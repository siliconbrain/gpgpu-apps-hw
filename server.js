const config = require('./config');
const express = require('express');
const nooocl = require('nooocl');
const R = require('ramda');
const ref = require('ref');

const host = new nooocl.CLHost(config.version);

const platform = host.getPlatforms()[config.platform];
console.log(`Platform selected:`, platform.name);

const devices = platform.allDevices();

const context = new nooocl.CLContext(devices);

const commandQueues = devices.map(device => new nooocl.CLCommandQueue(context, device));

const commonSource = `
size_t getPointIndex(size_t u, size_t v, size_t stride) {
    return (u % stride) + (v * stride);
}
`
const getSurfacePointsSource = `
struct Params {
    float phiMin;
    float phiStep;
    float zMin;
    float zStep;
};

float getRho(float phi, float z) {
    return fmin(fmax(sin((z + phi) * 2.0f), -0.7f), 0.7f) + 3.0f;
}

float3 toCartesian(float rho, float phi, float z) {
    return (float3)(cos(phi) * rho, sin(phi) * rho, z);
}

float3 getSurfacePoint(size_t u, size_t v, __constant struct Params *params) {
    float phi = params->phiMin + u * params->phiStep;
    float z = params->zMin + v * params->zStep;
    return toCartesian(getRho(phi, z), phi, z);
}
`
const dynamicProgramSource = getSurfacePointsSource + commonSource + `
__kernel void computePoints(__global float *points, __constant struct Params *params) {
    size_t stride = get_global_size(0);
    size_t u = get_global_id(0);
    size_t v = get_global_id(1);

    vstore3(getSurfacePoint(u, v, params), getPointIndex(u, v, stride), points);
}
`
const staticProgramSource = commonSource + `
size_t getQuadIndex(size_t u, size_t v, size_t stride) {
    return (u % stride) + (v * stride);
}

size_t getTriangleIndex(size_t u, size_t v, size_t w, size_t stride) {
    return getQuadIndex(u, v, stride) * 2 + w;
}

__kernel void computeMesh(__global const float *points, __global float *vertices, __global float *normals) {
    size_t stride = get_global_size(0);
    size_t u = get_global_id(0);
    size_t v = get_global_id(1);
    size_t w = get_global_id(2);

    float3 v0 = vload3(getPointIndex(u, v, stride), points);
    float3 v1 = vload3(getPointIndex(u + 1, v, stride) * (1 - w) + getPointIndex(u + 1, v + 1, stride) * w, points);
    float3 v2 = vload3(getPointIndex(u + 1, v + 1, stride) * (1 - w) + getPointIndex(u, v + 1, stride) * w, points);

    size_t ti = getTriangleIndex(u, v, w, stride);

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

function logBuildResult(buildResult) {
    console.log(`Build on ${buildResult.device.name}: ${isError(buildResult) ? '✗' : '✓'}`);
    if (isError(buildResult)) console.log(buildResult.log);
}

function setBuffer(buffer, ...data) {
    var offset = 0;
    data.forEach(({type, value}) => {
        type.set(buffer, offset, value);
        offset += type.size;
    });
    return buffer;
}

function getGeometry() {
    var z_min = 0.0, z_max = 10.0, z_res = 50;
    var phi_min = 0.0, phi_max = 2.0*Math.PI, phi_res = 60;

    var z_step = (z_max - z_min) / z_res;
    var phi_step = (phi_max - phi_min) / phi_res;

    const staticProgram = context.createProgram(staticProgramSource);
    const dynamicProgram = context.createProgram(dynamicProgramSource);
    return Promise.all([
        buildProgram(staticProgram).then(res => (console.log(`Static program built.`), res)),
        buildProgram(dynamicProgram).then(res => (console.log(`Dynamic program built.`), res))
    ]).then(([staticBuildResults, dynamicBuildResults]) => {
        if (staticBuildResults.some(isError) || dynamicBuildResults.some(isError)) {
            console.error('Some build(s) failed.');
            console.info('Static program results:');
            staticBuildResults.forEach(logBuildResult);
            console.info('Dynamic program results:');
            dynamicBuildResults.forEach(logBuildResult);
        } else {
            const computeMeshKernel = staticProgram.createKernel('computeMesh');
            const computePointsKernel = dynamicProgram.createKernel('computePoints');

            const params = new nooocl.CLBuffer(context, host.cl.defs.CL_MEM_READ_ONLY, ref.types.float.size * 4);
            const paramsBuffer = setBuffer(Buffer.alloc(ref.types.float.size * 4), ...[phi_min, phi_step, z_min, z_step].map(value => ({ type: ref.types.float, value})));

            const numOfPoints = (z_res + 1) * phi_res;
            const numOfVertices = z_res * phi_res * 2 * 3;
            const points = new nooocl.CLBuffer(context, host.cl.defs.CL_MEM_READ_WRITE, ref.types.float.size * 3 * numOfPoints);
            const vertices = new nooocl.CLBuffer(context, host.cl.defs.CL_MEM_WRITE_ONLY, ref.types.float.size * 3 * numOfVertices);
            const vertexBuffer = Buffer.alloc(vertices.size);
            const normals = new nooocl.CLBuffer(context, host.cl.defs.CL_MEM_WRITE_ONLY, ref.types.float.size * 3 * numOfVertices);
            const normalBuffer = Buffer.alloc(normals.size);
            console.log(`Buffers allocated.`, points.size, vertices.size, normals.size);

            computePointsKernel.setArgs(points, params);
            computeMeshKernel.setArgs(points, vertices, normals);
            console.log(`Kernel args set.`);
            
            return commandQueues[0].waitable().enqueueWriteBuffer(params, 0, params.size, paramsBuffer).promise.then(() => {
                console.log(`Params buffer write done.`);
                return commandQueues[0].waitable().enqueueNDRangeKernel(computePointsKernel, new nooocl.NDRange(phi_res, z_res + 1)).promise;
            }).then(() => {
                console.log(`Points computed.`);
                return commandQueues[0].waitable().enqueueNDRangeKernel(computeMeshKernel, new nooocl.NDRange(phi_res, z_res, 2)).promise;
            }).then(() => {
                console.log(`Mesh computed.`);
                return commandQueues[0].waitable().enqueueReadBuffer(vertices, 0, vertices.size, vertexBuffer).promise;
            }).then(() => {
                console.log(`Vertices read.`);
                return commandQueues[0].waitable().enqueueReadBuffer(normals, 0, normals.size, normalBuffer).promise;
            }).then(() => {
                console.log(`Normals read.`);
                return ({
                    vertices: new Array(numOfVertices * 3).fill(null).map((v, i) => vertexBuffer.readFloatLE(ref.types.float.size * i)),
                    normals: new Array(numOfVertices * 3).fill(null).map((v, i) => normalBuffer.readFloatLE(ref.types.float.size * i)),
                });
            });
        }
    });
}

const app = express();

//app.use(express.static('.'));

app.options('/geometry.json', function (req, res) {
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.set('Access-Control-Allow-Methods', 'GET');
    res.set('Access-Control-Allow-Origin', '*');
    res.sendStatus(200);
});
app.get('/geometry.json', function (req, res) {
    res.set('Access-Control-Allow-Origin', '*');
    res.set('Content-Disposition', 'attachment; filename="geometry.json"');
    getGeometry().then(geometry => res.json(geometry));
});

app.listen(config.port, function () {
    console.log(`Listening on http://localhost:${config.port}...`);
})