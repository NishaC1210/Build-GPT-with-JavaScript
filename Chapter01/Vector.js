function vector_add(v1, v2) {
    const result = new Array(v1.length);
    for (let i = 0; i < result.length; ++i) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

function vector_scale(vec, s) {
    const result = new Array(vec.length);
    for (let i = 0; i < result.length; ++i) {
        result[i] = vec[i] * s;
    }
    return result;
}

function vector_subtract(v1, v2) {
    const result = new Array(v1.length);
    for (let i = 0; i < result.length; ++i) {
        result[i] = v1[i] - v2[i];
    }
    return result;
}

function vector_magnitude(vec) {
    let sum = 0.0;
    for (let i = 0; i < vec.length; ++i) {
        sum += vec[i] * vec[i];
    }
    return Math.sqrt(sum);
}

function vector_normalized(vec) {
    const len = vector_magnitude(vec);
    const result = new Array(vec.length);
    for (let i = 0; i < result.length; ++i) {
        result[i] = vec[i] / len;
    }
    return result;
}

function vector_dot(v1, v2) {
    let result = 0.0;
    for (let i = 0; i < v1.length; ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

function vector_angle(v1, v2) {
    const lengths = vector_magnitude(v1) * vector_magnitude(v2);

    let cosine = vector_dot(v1, v2) / lengths;
    // clamp to -1 to 1 to avoid float error
    cosine = Math.min(1, Math.max(-1, cosine));
    return Math.acos(cosine);
}

function vector_softmax(vec) {
    let max = vec[0];
    for (let i = 1; i < vec.length; ++i) {
        if (vec[i] > max) {
            max = vec[i];
        }
    }
    const result = new Array(vec.length);
    let sum = 0.0;
    for (let i = 0; i < result.length; ++i) {
        result[i] = Math.exp(vec[i] - max);
        sum += result[i];
    }
    for (let i = 0; i < result.length; ++i) {
        result[i] /= sum;
    }
    return result;
}