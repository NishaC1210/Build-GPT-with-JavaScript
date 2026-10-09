import { Tensor } from './Tensor.js';

export class RMSNorm {
    constructor(feature_dim) {
        // Learned per-feature scale, applied after normalization. Starts at one,
        // so the layer begins as a plain normalize until training moves it.
        this.gamma = new Tensor(1, feature_dim);
        this.gamma.initToOnes();
    }

    forward(x) {
        const rows = x.rows;
        const n = x.columns;
        const eps = 1e-5;
        const gamma = this.gamma;
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
                // normalize, then scale by the per-feature gamma
                out.data[base + col] = x.data[base + col] * invRms * gamma.data[col];
            }
        }

        if (Tensor.gradEnabled) {
            out._inputs = [x, gamma];
            out._backward = () => {
                for (let row = 0; row < rows; ++row) {
                    const base = row * n;
                    const invRms = invRmsByRow[row];
                    const invRms3 = invRms * invRms * invRms;

                    // Upstream gradient passes through gamma before the RMS rule.
                    let dotGX = 0.0;
                    for (let col = 0; col < n; ++col) {
                        const gNorm = out.grad[base + col] * gamma.data[col];
                        dotGX += gNorm * x.data[base + col];
                    }
                    for (let col = 0; col < n; ++col) {
                        const g = out.grad[base + col];
                        const xv = x.data[base + col];
                        const gNorm = g * gamma.data[col];
                        x.grad[base + col] += gNorm * invRms - (xv * dotGX * invRms3) / n;
                        // gamma is shared across rows: sum the normalized input times upstream grad
                        gamma.grad[col] += xv * invRms * g;
                    }
                }
            };
        }
        return out;
    }

    parameters() {
        return [this.gamma];
    }
}

export class MultiHeadAttention {
    constructor(feature_dim, num_heads, rope_base) {
        this.num_heads = num_heads;
        this.head_dim = feature_dim / num_heads; // must divide evenly, and be even for RoPE
        this.rope_base = rope_base;

        // Each head owns Q, K, V down to head_dim, and an output projection
        // back up to feature_dim. The summed outputs equal a concat plus one
        // big output projection, so no concat operation is needed.
        this.heads = [];
        for (let i = 0; i < num_heads; ++i) {
            const learnedQ = new Tensor(feature_dim, this.head_dim);
            learnedQ.initToSmallRandom();

            const learnedK = new Tensor(feature_dim, this.head_dim);
            learnedK.initToSmallRandom();

            const learnedV = new Tensor(feature_dim, this.head_dim);
            learnedV.initToSmallRandom();

            const learnedO = new Tensor(this.head_dim, feature_dim);
            learnedO.initToSmallRandom();

            this.heads.push({ learnedQ, learnedK, learnedV, learnedO });
        }
    }

    forward(rms_norm_matrix, cache = null, layerIndex = 0) { // Shape: new_tokens, feature_dim
        const scale = 1.0 / Math.sqrt(this.head_dim);
        let output = null;

        for (let i = 0; i < this.num_heads; ++i) {
            const head = this.heads[i];

            let Q = rms_norm_matrix.matmul(head.learnedQ); // Shape: new_tokens, head_dim
            let K = rms_norm_matrix.matmul(head.learnedK); // Shape: new_tokens, head_dim
            let V = rms_norm_matrix.matmul(head.learnedV); // Shape: new_tokens, head_dim

            // The new tokens sit after whatever this head has already cached.
            const pastK = cache === null ? null : cache.keys[layerIndex][i];
            const pastV = cache === null ? null : cache.values[layerIndex][i];
            const positionOffset = pastK === null ? 0 : pastK.rows;

            // Rotate by true position. Cached keys keep the rotation they were
            // stored with; the new keys are rotated from the offset onward.
            Q = Q.ropeRotated(this.rope_base, positionOffset);
            K = K.ropeRotated(this.rope_base, positionOffset);

            // Grow the cache with the new keys and values, then attend over the
            // full history. Without a cache, K and V are just the new tokens.
            if (cache !== null) {
                K = pastK === null ? K : pastK.appendRow(K);
                V = pastV === null ? V : pastV.appendRow(V);
                cache.keys[layerIndex][i] = K;
                cache.values[layerIndex][i] = V;
            }

            let scores = Q.matmul(K.transposed()); // Shape: new_tokens, total_tokens
            scores = scores.scale(scale);

            // The prompt's first pass scores a square block and needs the mask.
            // Every later token sees only its past, so no mask is needed.
            if (positionOffset === 0) {
                scores = scores.causalMasked();
            }

            const probabilities = scores.softMaxedRows();
            const mixed = probabilities.matmul(V);         // Shape: new_tokens, head_dim

            // Project this head up to feature_dim and accumulate.
            const projected = mixed.matmul(head.learnedO); // Shape: new_tokens, feature_dim
            output = output === null ? projected : output.add(projected);
        }

        return output; // Shape: new_tokens, feature_dim
    }

    parameters() {
        const params = [];
        for (let i = 0; i < this.heads.length; ++i) {
            const head = this.heads[i];
            const matrices = [head.learnedQ, head.learnedK, head.learnedV, head.learnedO];
            for (let j = 0; j < matrices.length; ++j) {
                const matParams = matrices[j].parameters();
                for (let k = 0; k < matParams.length; ++k) {
                    params.push(matParams[k]);
                }
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

        // Per-neuron bias on the hidden layer, added to each pre-activation
        // before GELU. Starts at zero, so the layer begins as it did without it.
        this.bias = new Tensor(1, hidden_dim_size);
        this.bias.initToZeroes();
    }

    forward(x) { // Shape: sequence_length, feature_dim
        const up = x.matmul(this.learnedUp); // Shape: sequence_length, hidden_dim

        // Shift each pre-activation by its neuron's bias, broadcast across
        // every row. Fused into one node so the shared bias gradient sums
        // down each column in a single pass.
        const bias = this.bias;
        const rows = up.rows;
        const cols = up.columns;
        const preActivation = new Tensor(rows, cols);
        for (let row = 0; row < rows; ++row) {
            const base = row * cols;
            for (let col = 0; col < cols; ++col) {
                preActivation.data[base + col] = up.data[base + col] + bias.data[col];
            }
        }

        if (Tensor.gradEnabled) {
            preActivation._inputs = [up, bias];
            preActivation._backward = () => {
                for (let row = 0; row < rows; ++row) {
                    const base = row * cols;
                    for (let col = 0; col < cols; ++col) {
                        const g = preActivation.grad[base + col];
                        up.grad[base + col] += g;
                        // bias is shared by every row, so its gradient sums down the column.
                        bias.grad[col] += g;
                    }
                }
            };
        }

        // Bend with GELU, then project back down.
        return preActivation.gelu().matmul(this.learnedDown); // Shape: sequence_length, feature_dim
    }

    parameters() {
        const params = [];
        const tensors = [this.learnedUp, this.learnedDown, this.bias];
        for (let i = 0; i < tensors.length; ++i) {
            const sub = tensors[i].parameters();
            for (let j = 0; j < sub.length; ++j) {
                params.push(sub[j]);
            }
        }
        return params;
    }
}

export class TransformerBlock {
    constructor(feature_dim, num_heads, rope_base) {
        // MLP usually expands to 4x the feature dimension.
        const hidden_dim = feature_dim * 4;

        this.attentionNorm = new RMSNorm(feature_dim);
        this.attention = new MultiHeadAttention(feature_dim, num_heads, rope_base);

        this.mlpNorm = new RMSNorm(feature_dim);
        this.mlp = new MLP(feature_dim, hidden_dim);
    }

    forward(x, cache = null, layerIndex = 0) { // cache and layerIndex are new
        // Attention sub-layer, with residual. The cache and layer index pass
        // straight through to attention; the MLP and norms never mix tokens,
        // so they have nothing to cache.
        const attentionInput = this.attentionNorm.forward(x);
        const attentionDelta = this.attention.forward(attentionInput, cache, layerIndex);
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
