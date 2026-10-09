import { Tensor } from './Tensor.js';

export class RMSNorm {
    // RMSNorm is the first layer that builds a tensor operation itself. The
    // forward pass computes one inverse root mean square per row and multiplies
    // every cell in that row by it. The inverse is cached in invRmsByRow,
    // because the backward pass needs the same scale again.
    forward(x) {
        const rows = x.rows;
        const n = x.columns;
        const eps = 1e-5;
        const out = new Tensor(rows, n);
        const invRmsByRow = new Float32Array(rows); // cached for the backward

        for (let row = 0; row < rows; ++row) {
            const base = row * n;

            let sumSq = 0.0;
            for (let col = 0; col < n; ++col) {
                const v = x.data[base + col];
                sumSq += v * v;
            }

            const invRms = 1 / Math.sqrt(sumSq / n + eps);
            invRmsByRow[row] = invRms;

            for (let col = 0; col < n; ++col) {
                out.data[base + col] = x.data[base + col] * invRms;
            }
        }

        if (Tensor.gradEnabled) {
            out._inputs = [x];
            out._backward = () => {
                // dx_a = g_a / r  -  x_a * (sum_k g_k x_k) / (n * r^3)
                for (let row = 0; row < rows; ++row) {
                    const base = row * n;
                    const invRms = invRmsByRow[row];
                    const invRms3 = invRms * invRms * invRms;

                    let dotGX = 0.0;
                    for (let col = 0; col < n; ++col) {
                        dotGX += out.grad[base + col] * x.data[base + col];
                    }
                    for (let col = 0; col < n; ++col) {
                        const g = out.grad[base + col];
                        const xv = x.data[base + col];
                        x.grad[base + col] += g * invRms - (xv * dotGX * invRms3) / n;
                    }
                }
            };
        }
        return out;
    }

    parameters() {
        return []; // no learned values yet
    }
}

export class AttentionHead {
    constructor(feature_dim, rope_base) {
        this.head_dim = feature_dim;
        this.rope_base = rope_base; // New

        this.learnedQ = new Tensor(feature_dim, this.head_dim);
        this.learnedQ.initToSmallRandom();

        this.learnedK = new Tensor(feature_dim, this.head_dim);
        this.learnedK.initToSmallRandom();

        this.learnedV = new Tensor(feature_dim, this.head_dim);
        this.learnedV.initToSmallRandom();
    }

    forward(rms_norm_matrix) { // Shape: sequence_length, feature_dim
        let Q = rms_norm_matrix.matmul(this.learnedQ); // Shape: sequence_length, head_dim
        let K = rms_norm_matrix.matmul(this.learnedK); // Shape: sequence_length, head_dim
        const V = rms_norm_matrix.matmul(this.learnedV); // Shape: sequence_length, head_dim

        Q = Q.ropeRotated(this.rope_base); // queries rotated by position
        K = K.ropeRotated(this.rope_base); // keys rotated by position
        // values left untouched, not rotated

        let scores = Q.matmul(K.transposed()); // Shape: sequence_length, sequence_length
        const scale = 1.0 / Math.sqrt(this.head_dim); // multiply by reciprocal instead of dividing
        scores = scores.scale(scale); // Element-wise scalar multiplication

        scores = scores.causalMasked(); // Upper-right triangle is now -Infinity
        const probabilities = scores.softMaxedRows(); // Shape: sequence_length, sequence_length
        const mixed = probabilities.matmul(V); // Shape: sequence_length, head_dim

        // head_dim == feature_dim, no output projection needed
        return mixed; // Output shape same as input shape
    }

    parameters() {
        const params = [];
        const matrices = [this.learnedQ, this.learnedK, this.learnedV];
        for (let i = 0; i < matrices.length; ++i) {
            const matParams = matrices[i].parameters();
            for (let j = 0; j < matParams.length; ++j) {
                params.push(matParams[j]);
            }
        }
        return params;
    }
}

export class MLP {
    constructor(feature_dim_size, hidden_dim_size) {
        this.learnedUp = new Tensor(feature_dim_size, hidden_dim_size);
        this.learnedUp.initToSmallRandom();

        this.learnedDown = new Tensor(hidden_dim_size, feature_dim_size);
        this.learnedDown.initToSmallRandom();
    }

    forward(x) { // Shape: sequence_length, feature_dim
        // Project up, bend with ReLU, project back down — now one chain of
        // tensor operations, since ReLU is a tensor op.
        return x.matmul(this.learnedUp).relu().matmul(this.learnedDown);
    }

    parameters() {
        const params = [];
        const matrices = [this.learnedUp, this.learnedDown];
        for (let i = 0; i < matrices.length; ++i) {
            const matParams = matrices[i].parameters();
            for (let j = 0; j < matParams.length; ++j) {
                params.push(matParams[j]);
            }
        }
        return params;
    }
}

export class TransformerBlock {
    constructor(feature_dim, rope_base) {
        // MLP usually expands to 4x the feature dimension.
        const hidden_dim = feature_dim * 4;

        this.attentionNorm = new RMSNorm();
        this.attention = new AttentionHead(feature_dim, rope_base);

        this.mlpNorm = new RMSNorm();
        this.mlp = new MLP(feature_dim, hidden_dim);
    }

    forward(x) { // Shape: sequence_length, feature_dim
        // Attention sub-layer, with residual
        const attentionInput = this.attentionNorm.forward(x);
        const attentionDelta = this.attention.forward(attentionInput);
        const afterAttention = x.add(attentionDelta);

        // MLP sub-layer, with residual
        const mlpInput = this.mlpNorm.forward(afterAttention);
        const mlpDelta = this.mlp.forward(mlpInput);
        const afterMLP = afterAttention.add(mlpDelta);

        return afterMLP;
    }

    parameters() {
        const params = [];
        const components = [this.attentionNorm, this.attention, this.mlpNorm, this.mlp];
        for (let i = 0; i < components.length; ++i) {
            const sub = components[i].parameters();
            for (let j = 0; j < sub.length; ++j) {
                params.push(sub[j]);
            }
        }
        return params;
    }
}
