var DEBUG = true;

vec2.angle = function(v) {
    return Math.atan2(v[1], v[0]);
};

mat4.uniformScale = function(out, m, s) {
    return mat4.scale(out, m, [s, s, s]);
};

function updateCanvasBackBufferSize(gl, scale=1) {
    var canvas = gl.canvas;
    var width = canvas.clientWidth / scale;
    var height = canvas.clientHeight / scale;

    if (canvas.width != width || canvas.height != height) {
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, width, height);
    }
}

function createGL(canvas) {
    return canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
}

function createShader(gl, type, source) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        return shader;
    } else {
        console.error(gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
    }
}

function createProgram(gl, vertexShader, fragmentShader) {
    var program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (gl.getProgramParameter(program, gl.LINK_STATUS)) {
        return program;
    } else {
        console.error(gl.getProgramInfoLog(program));
        gl.deleteProgram(program);
    }
}

function clamp(min, val, max) {
    return Math.max(min, Math.min(max, val));
}

function getGeometry(params) {
    return fetch(`/geometry.json`, {
        body: JSON.stringify(params),
        headers: {
            'Content-Type': 'application/json'
        },
        method: 'POST',
        mode: 'cors',
    }).then(response => response.json());
}

function init() {
    var canvas = document.querySelector('canvas');
    var vertexShaderSource = document.getElementById('vertex-shader').innerText;
    var fragmentShaderSource = document.getElementById('fragment-shader').innerText;
    var bgColor = [1, 1, 1, 1];

    var gl = createGL(canvas);
    if (DEBUG) window.gl = gl;

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);

    updateCanvasBackBufferSize(gl);

    var vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    var fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
    var program = createProgram(gl, vertexShader, fragmentShader);
    gl.useProgram(program);

    var positionAttributeLocation = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(positionAttributeLocation);

    var normalAttributeLocation = gl.getAttribLocation(program, "a_normal");
    gl.enableVertexAttribArray(normalAttributeLocation);

    var modelMatrix = mat4.create();
    var viewMatrix = mat4.fromTranslation([], [0, 0, -1]);
    var projectionMatrix = mat4.perspective([], Math.PI / 3, canvas.width / canvas.height, 0.01, 100);

    var pvmMatrixLocation = gl.getUniformLocation(program, "u_pvmMatrix");
    var vmitMatrixLocation = gl.getUniformLocation(program, "u_vmitMatrix");

    var coordVertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, coordVertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]), gl.STATIC_DRAW);

    var vertexBuffer = gl.createBuffer();
    var normalBuffer = gl.createBuffer();

    var draw = function() {};

    function loadGeometry(geometry) {
        if (DEBUG) window.geometry = geometry;

        if (geometry.normals.length !== geometry.vertices.length) console.error('|ns| != |vs|');
        if (geometry.vertices.length % 3 !== 0) console.error('|vs| % 3 != 0');

        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(geometry.vertices), gl.STREAM_DRAW);
        gl.vertexAttribPointer(positionAttributeLocation, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(geometry.normals), gl.STREAM_DRAW);
        gl.vertexAttribPointer(normalAttributeLocation, 3, gl.FLOAT, false, 0, 0);

        return geometry;
    }

    window.addEventListener('resize', function() {
        updateCanvasBackBufferSize(gl);
        projectionMatrix = mat4.perspective([], Math.PI / 3, canvas.width / canvas.height, 0.01, 100);
        draw();
    });

    gl.clearColor(...bgColor);

    canvas.addEventListener('mousemove', function (e) {
        var scaleFactor = 2;
        var sphereSize = 0.7;
        if (e.buttons === 1) {
            var endV = vec2.fromValues(
                2.0 * e.clientX / e.target.clientWidth - 1.0,
                -2.0 * e.clientY / e.target.clientHeight + 1.0
            );
            var startV = vec2.fromValues(
                2.0 * (e.clientX - e.movementX) / e.target.clientWidth - 1.0,
                -2.0 * (e.clientY - e.movementY) / e.target.clientHeight + 1.0
            );
            var startL = vec2.length(startV);
            if (startL < sphereSize) {
                var startVL = vec2.scale([], startV, 1 / sphereSize);
                var startD = [startVL[0], startVL[1], Math.cos(Math.asin(clamp(-1.0, vec2.length(startVL), 1.0)))];
                var endVL = vec2.scale([], endV, 1 / sphereSize);
                var endD = [endVL[0], endVL[1], Math.cos(Math.asin(clamp(-1.0, vec2.length(endVL), 1.0)))];
                var axis = [];
                var angle = quat.getAxisAngle(axis, quat.rotationTo([], startD, endD));
                vec3.transformMat4(axis, axis, mat4.invert([], modelMatrix));
                mat4.rotate(modelMatrix, modelMatrix, angle, axis);
            } else {
                var radial = vec2.length(endV) - startL;
                var angle = vec2.angle(endV) - vec2.angle(startV);
                mat4.rotate(modelMatrix, modelMatrix, angle, vec3.transformMat4([], [0, 0, 1], mat4.invert([], modelMatrix)));
                mat4.uniformScale(viewMatrix, viewMatrix, 1.0 + radial * scaleFactor);
            }
            draw();
        }
    });

    document.querySelectorAll('input[name="primitive-type"]').forEach(elem => elem.addEventListener('change', () => {
        draw();
    }));

    document.getElementById('submit').addEventListener('click', function() {
        getGeometry({
            u: {
                min: parseFloat(document.getElementById('u-min').value),
                max: parseFloat(document.getElementById('u-max').value),
                res: parseInt(document.getElementById('u-res').value)
            },
            v: {
                min: parseFloat(document.getElementById('v-min').value),
                max: parseFloat(document.getElementById('v-max').value),
                res: parseInt(document.getElementById('v-res').value)
            },
            expr: document.getElementById('expr').value,
            conv: document.getElementById('conv').value,
        }).then(loadGeometry).then(function (geometry) {
            var size = R.splitEvery(3, geometry.vertices).map(v => vec3.length(v)).reduce(R.max, 0);
            viewMatrix = mat4.uniformScale([], mat4.fromTranslation([], [0, 0, -1]), 0.5 / size);
            draw = function() {
                var viewModelMatrix = mat4.mul([], viewMatrix, modelMatrix);
                gl.uniformMatrix4fv(pvmMatrixLocation, false, mat4.mul([], projectionMatrix, viewModelMatrix));
                gl.uniformMatrix4fv(vmitMatrixLocation, false, mat4.transpose([], mat4.invert([], viewModelMatrix)));
                gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                var primitiveType = [...document.querySelectorAll('input[name="primitive-type"]')].find(elem => elem.checked).value;
                gl.drawArrays(gl[primitiveType], 0, geometry.vertices.length / 3);
            }

            draw();
        }, err => console.log(err));
    });
    document.getElementById('reset-view-matrix').addEventListener('click', function() {
        viewMatrix = mat4.uniformScale([], mat4.fromTranslation([], [0, 0, -1]), 0.1);
        draw();
    });
}
