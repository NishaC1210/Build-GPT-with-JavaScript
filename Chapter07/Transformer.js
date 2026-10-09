import { Value } from './Value.js';
import { Matrix } from './Matrix.js';

export class RMSNorm {
    forward(matrix) {
        const RMS_EPSILON = 1e-5;
        const sequenceLength = matrix.rows;
        const featureDimensions = matrix.columns;

        const result = new Matrix(sequenceLength, featureDimensions);
        // No init — every cell is assigned below.

        for (let row = 0; row < sequenceLength; ++row) {
            // 1. Sum of squares for this row, built as a graph node.
            let sumSq = new Value(0.0);
            for (let col = 0; col < featureDimensions; ++col) {
                const v = matrix.get(row, col);
                sumSq = sumSq.add(v.mul(v));
            }

            // 2. rms = sqrt(meanSq + eps)
            const meanSq = sumSq.div(featureDimensions);
            const rms = meanSq.add(RMS_EPSILON).sqrt();
            const invRms = rms.pow(-1); // reciprocal as one shared node

            // 3. Normalize: multiply each component by 1/rms.
            for (let col = 0; col < featureDimensions; ++col) {
                result.set(row, col, matrix.get(row, col).mul(invRms));
            }
        }

        return result;
    }

    parameters() {
        return [];
    }
}

export class AttentionHead {
    constructor(feature_dim) {
        this.head_dim = feature_dim;

        this.learnedQ = new Matrix(feature_dim, this.head_dim);
        this.learnedQ.initToSmallRandom();

        this.learnedK = new Matrix(feature_dim, this.head_dim);
        this.learnedK.initToSmallRandom();

        this.learnedV = new Matrix(feature_dim, this.head_dim);
        this.learnedV.initToSmallRandom();
    }

    forward(rms_norm_matrix) { // Shape: sequence_length, feature_dim
        const Q = rms_norm_matrix.matmul(this.learnedQ); // Shape: sequence_length, head_dim
        const K = rms_norm_matrix.matmul(this.learnedK); // Shape: sequence_length, head_dim
        const V = rms_norm_matrix.matmul(this.learnedV); // Shape: sequence_length, head_dim

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
        this.hiddenDim = hidden_dim_size;

        this.learnedUp = new Matrix(feature_dim_size, hidden_dim_size);
        this.learnedUp.initToSmallRandom();

        this.learnedDown = new Matrix(hidden_dim_size, feature_dim_size);
        this.learnedDown.initToSmallRandom();
    }

    activation(value) { // ReLU, now on a Value
        return value.relu();
    }

    forward(matrix) { // Shape: sequence_length, feature_dim
        // Project to hidden dim size
        let hidden = matrix.matmul(this.learnedUp); // Shape: sequence_length, hidden_dim

        // Bend: apply the activation to every value in the hidden matrix
        const sequenceLength = matrix.rows;
        for (let row = 0; row < sequenceLength; ++row) {
            for (let col = 0; col < this.hiddenDim; ++col) {
                hidden.set(row, col, this.activation(hidden.get(row, col)));
            }
        }

        // Project back down to feature dim size
        const out = hidden.matmul(this.learnedDown); // Shape: sequence_length, feature_dim
        return out;
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
    constructor(feature_dim) {
        // MLP usually expands to 4x the feature dimension.
        const hidden_dim = feature_dim * 4;

        this.attentionNorm = new RMSNorm();
        this.attention = new AttentionHead(feature_dim);

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