const bodyParser = require('body-parser');
const config = require('./config');
const express = require('express');
const nooocl = require('nooocl');
const R = require('ramda');
const ref = require('ref');

const app = express();

app.use(express.static('static'));

app.use(bodyParser.json());

app.options('/geometry.json', function (req, res) {
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.set('Access-Control-Allow-Methods', 'GET');
    res.set('Access-Control-Allow-Origin', '*');
    res.sendStatus(200);
});



const host = new nooocl.CLHost(config.version);

const platform = host.getPlatforms()[config.platform];
console.log(`Platform selected:`, platform.name);

const devices = platform.allDevices();

const device = devices[0];
console.log(`Device selected:`, device.name);

const context = new nooocl.CLContext(device);

const commandQueue = new nooocl.CLCommandQueue(context, device);

function getDynamicProgramSource({expr, conv}) {
    return `
float getValueAt(float u, float v) {
    return ${expr || '0.0f'};
}

float3 toCartesian(float u, float v, float w) {
    return (float3)(${conv || 'u, v, w'});
}

struct Params {
    float uMin;
    float uStep;
    float vMin;
    float vStep;
};

float3 getSurfacePoint(size_t ui, size_t vi, __constant struct Params *params) {
    float u = params->uMin + ui * params->uStep;
    float v = params->vMin + vi * params->vStep;
    return toCartesian(u, v, getValueAt(u, v));
}

size_t getPointIndex(size_t ui, size_t vi, size_t stride) {
    return ui + vi * stride;
}

__kernel void computePoints(__global float *points, __constant struct Params *params) {
    size_t stride = get_global_size(0);
    size_t ui = get_global_id(0);
    size_t vi = get_global_id(1);

    vstore3(getSurfacePoint(ui, vi, params), getPointIndex(ui, vi, stride), points);
}
`;
}
const staticProgramSource = `
size_t getPointIndex(size_t ui, size_t vi, size_t stride) {
    return ui + vi * (stride + 1);
}

size_t getQuadIndex(size_t ui, size_t vi, size_t stride) {
    return ui + vi * stride;
}

size_t getTriangleIndex(size_t ui, size_t vi, size_t wi, size_t stride) {
    return getQuadIndex(ui, vi, stride) * 2 + wi;
}

__kernel void computeMesh(__global const float *points, __global float *vertices, __global float *normals) {
    size_t stride = get_global_size(0);
    size_t ui = get_global_id(0);
    size_t vi = get_global_id(1);
    size_t wi = get_global_id(2);

    float3 v0 = vload3(getPointIndex(ui, vi, stride), points);
    float3 v1 = vload3(getPointIndex(ui + 1, vi, stride) * (1 - wi) + getPointIndex(ui + 1, vi + 1, stride) * wi, points);
    float3 v2 = vload3(getPointIndex(ui + 1, vi + 1, stride) * (1 - wi) + getPointIndex(ui, vi + 1, stride) * wi, points);

    size_t ti = getTriangleIndex(ui, vi, wi, stride);

    vstore3(v0, 3 * ti + 0, vertices);
    vstore3(v1, 3 * ti + 1, vertices);
    vstore3(v2, 3 * ti + 2, vertices);

    float3 n = normalize(cross(v1 - v0, v2 - v0));

    vstore3(n, 3 * ti + 0, normals);
    vstore3(n, 3 * ti + 1, normals);
    vstore3(n, 3 * ti + 2, normals);
}
`

function getBuildResults(program) {
    return devices.map(device => ({
        device,
        program,
        status: program.getBuildStatus(device),
        log: program.getBuildLog(device),
    }));
}

function buildProgram(program) {
    return program.build().then(() => getBuildResults(program));
}

const isError = (buildResult) => buildResult.status === host.cl.defs.CL_BUILD_ERROR;

function logBuildResults(buildResults) {
    console.info(`Build results:`);
    buildResults.forEach(buildResult => {
        console.info(`Build on ${buildResult.device.name}: ${isError(buildResult) ? '✗' : '✓'}`);
        if (isError(buildResult)) console.log(buildResult.log);
    });
}

function setBuffer(buffer, ...data) {
    var offset = 0;
    data.forEach(({type, value}) => {
        type.set(buffer, offset, value);
        offset += type.size;
    });
    return buffer;
}

function getStep({min, max, res}) {
    return (max - min) / res;
}

function readFloats(buffer) {
    if (buffer.length % ref.types.float.size !== 0) console.warn(`Buffer size not a multiple of float size.`);
    return new Array(buffer.length / ref.types.float.size).fill(null).map((v, i) => buffer.readFloatLE(ref.types.float.size * i));
}

const staticProgram = context.createProgram(staticProgramSource);
buildProgram(staticProgram).then(staticBuildResults => {
    console.log(`Static program build finished.`);
    if (staticBuildResults.some(isError)) {
        console.error(`Some build(s) failed.`);
        logBuildResults(staticBuildResults);
        throw new Error(`Static program build failure.`);
    } else {
        function getGeometry({u, v, expr, conv}) {
            var uStep = getStep(u);
            var vStep = getStep(v);

            const dynamicProgram = context.createProgram(getDynamicProgramSource({expr, conv}));
            return buildProgram(dynamicProgram).then(dynamicBuildResults => {
                console.log(`Dynamic program build finished.`);
                if (dynamicBuildResults.some(isError)) {
                    console.error(`Some build(s) failed.`);
                    logBuildResults(dynamicBuildResults);
                } else {
                    const numOfPoints = (u.res + 1) * (v.res + 1);
                    const points = new nooocl.CLBuffer(context, host.cl.defs.CL_MEM_READ_WRITE, ref.types.float.size * 3 * numOfPoints);

                    const numOfFloatsInVertex = 3;
                    const numOfVerticesInTriangle = 3;
                    const numOfTrianglesInPatch = 2;
                    const numOfPatches = u.res * v.res;
                    const numOfVertices = numOfPatches * numOfTrianglesInPatch * numOfVerticesInTriangle;
                    const vertices = new nooocl.CLBuffer(context, host.cl.defs.CL_MEM_WRITE_ONLY, ref.types.float.size * numOfFloatsInVertex * numOfVertices);
                    const vertexBuffer = Buffer.alloc(vertices.size);
                    const normals = new nooocl.CLBuffer(context, host.cl.defs.CL_MEM_WRITE_ONLY, ref.types.float.size * numOfFloatsInVertex * numOfVertices);
                    const normalBuffer = Buffer.alloc(normals.size);
                    console.log(`Buffers allocated.`);

                    const paramsBuffer = setBuffer(Buffer.alloc(ref.types.float.size * 4), ...[u.min, uStep, v.min, vStep].map(value => ({ type: ref.types.float, value})));
                    const params = new nooocl.CLBuffer(context, host.cl.defs.CL_MEM_READ_ONLY, paramsBuffer.length);
                    return commandQueue.waitable().enqueueWriteBuffer(params, 0, params.size, paramsBuffer).promise.then(() => {
                        console.log(`Params buffer write done.`);
                        const computePointsKernel = dynamicProgram.createKernel('computePoints');
                        computePointsKernel.setArgs(points, params);
                        return commandQueue.waitable().enqueueNDRangeKernel(computePointsKernel, new nooocl.NDRange(u.res + 1, v.res + 1)).promise.then(() => {
                            console.log(`Points computed.`);
                            computePointsKernel.release();
                            params.release();
                        });
                    }).then(() => {
                        const computeMeshKernel = staticProgram.createKernel('computeMesh');
                        computeMeshKernel.setArgs(points, vertices, normals);
                        return commandQueue.waitable().enqueueNDRangeKernel(computeMeshKernel, new nooocl.NDRange(u.res, v.res, 2)).promise.then(() => {
                            console.log(`Mesh computed.`);
                            computeMeshKernel.release();
                            points.release();
                        });
                    }).then(() => commandQueue.waitable().enqueueReadBuffer(vertices, 0, vertices.size, vertexBuffer).promise.then(() => {
                        console.log(`Vertices read.`);
                        vertices.release();
                    })).then(() => commandQueue.waitable().enqueueReadBuffer(normals, 0, normals.size, normalBuffer).promise.then(() => {
                        console.log(`Normals read.`);
                        normals.release();
                    })).then(() => ({
                        vertices: readFloats(vertexBuffer),
                        normals: readFloats(normalBuffer),
                    }));
                }
            });
        }

        app.post('/geometry.json', function (req, res) {
            res.set('Access-Control-Allow-Origin', '*');
            res.set('Content-Disposition', 'attachment; filename="geometry.json"');
            getGeometry(req.body).then(geometry => res.json(geometry));
        });

        app.listen(config.port, function () {
            console.log(`Listening on http://localhost:${config.port}...`);
        })

    }
}).catch(err => {
    console.error(err);
    process.exit(1);
});


