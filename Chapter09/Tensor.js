let randomState = 42;

function random() {
    randomState =
        (1664525 * randomState + 1013904223) % 4294967296;
    return randomState / 4294967296;
}

export class Tensor {
    // When false, operations skip building the graph:
    static gradEnabled = true;

    constructor(rows, columns) {
        this.rows = rows;
        this.columns = columns;
        this.data = new Float32Array(rows * columns); // zero-filled
        this.grad = null;             // allocated on demand by backward()
        this._inputs = [];            // tensors this one was built from
        this._backward = () => {};    // leaves do nothing
    }

    get(row, column) { // returns a plain number now, not a Value
        return this.data[row * this.columns + column];
    }

    set(row, column, value) {
        this.data[row * this.columns + column] = value;
    }

    // Initialization: these create the model's leaf parameters.

    initToZeroes() {
        for (let i = 0; i < this.data.length; ++i) {
            this.data[i] = 0.0;
        }
    }

    initToOnes() {
        for (let i = 0; i < this.data.length; ++i) {
            this.data[i] = 1.0;
        }
    }

    initToSmallRandom() {
        const scale = 1 / Math.sqrt(this.rows);
        for (let i = 0; i < this.data.length; ++i) {
            this.data[i] = (random() - 0.5) * 2 * scale; // uniform in [-scale, +scale)
        }
    }

    zeroGrad() {
        if (this.grad !== null) this.grad.fill(0);
    }

    parameters() {
        return [this]; // a tensor is one parameter block
    }

    // Element-wise operations (both tensors share the same shape).

    add(other) {
        const out = new Tensor(this.rows, this.columns);
        for (let i = 0; i < this.data.length; ++i) {
            out.data[i] = this.data[i] + other.data[i];
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this, other];
            out._backward = () => {
                for (let i = 0; i < this.data.length; ++i) {
                    this.grad[i]  += out.grad[i];
                    other.grad[i] += out.grad[i];
                }
            };
        }
        return out;
    }

    sub(other) {
        const out = new Tensor(this.rows, this.columns);
        for (let i = 0; i < this.data.length; ++i) {
            out.data[i] = this.data[i] - other.data[i];
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this, other];
            out._backward = () => {
                for (let i = 0; i < this.data.length; ++i) {
                    this.grad[i]  += out.grad[i];
                    other.grad[i] -= out.grad[i];
                }
            };
        }
        return out;
    }

    mul(other) { // element-wise multiply (matching numpy/pytorch); matrix multiply is matmul
        const out = new Tensor(this.rows, this.columns);
        for (let i = 0; i < this.data.length; ++i) {
            out.data[i] = this.data[i] * other.data[i];
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this, other];
            out._backward = () => {
                for (let i = 0; i < this.data.length; ++i) {
                    this.grad[i]  += other.data[i] * out.grad[i];
                    other.grad[i] += this.data[i]  * out.grad[i];
                }
            };
        }
        return out;
    }

    scale(scalar) { // multiply every value by one plain number
        const out = new Tensor(this.rows, this.columns);
        for (let i = 0; i < this.data.length; ++i) {
            out.data[i] = this.data[i] * scalar;
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                for (let i = 0; i < this.data.length; ++i) {
                    this.grad[i] += scalar * out.grad[i];
                }
            };
        }
        return out;
    }

    relu() {
        const out = new Tensor(this.rows, this.columns);
        for (let i = 0; i < this.data.length; ++i) {
            out.data[i] = this.data[i] > 0 ? this.data[i] : 0;
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                for (let i = 0; i < this.data.length; ++i) {
                    this.grad[i] += (this.data[i] > 0 ? 1 : 0) * out.grad[i];
                }
            };
        }
        return out;
    }

    // Matrix multiplication: the one node that replaces the most scalars.

    matmul(other) { // (this.rows x this.columns) * (other.rows x other.columns)
        const m = this.rows;
        const k = this.columns; // shared inner dimension
        const n = other.columns;
        const out = new Tensor(m, n);

        for (let row = 0; row < m; ++row) {
            for (let col = 0; col < n; ++col) {
                let sum = 0.0; // accumulate in double, store as float
                for (let element = 0; element < k; ++element) {
                    sum += this.data[row * k + element] * other.data[element * n + col];
                }
                out.data[row * n + col] = sum;
            }
        }

        if (Tensor.gradEnabled) {
            out._inputs = [this, other];
            out._backward = () => {
                // dThis = dOut * other^T
                for (let row = 0; row < m; ++row) {
                    for (let p = 0; p < k; ++p) {
                        let sum = 0.0;
                        for (let col = 0; col < n; ++col) {
                            sum += out.grad[row * n + col] * other.data[p * n + col];
                        }
                        this.grad[row * k + p] += sum;
                    }
                }
                // dOther = this^T * dOut
                for (let p = 0; p < k; ++p) {
                    for (let col = 0; col < n; ++col) {
                        let sum = 0.0;
                        for (let row = 0; row < m; ++row) {
                            sum += this.data[row * k + p] * out.grad[row * n + col];
                        }
                        other.grad[p * n + col] += sum;
                    }
                }
            };
        }
        return out;
    }

    transposed() {
        const out = new Tensor(this.columns, this.rows);
        for (let row = 0; row < this.rows; ++row) {
            for (let col = 0; col < this.columns; ++col) {
                out.data[col * this.rows + row] = this.data[row * this.columns + col];
            }
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                for (let row = 0; row < this.rows; ++row) {
                    for (let col = 0; col < this.columns; ++col) {
                        this.grad[row * this.columns + col] += out.grad[col * this.rows + row];
                    }
                }
            };
        }
        return out;
    }

    // Fused operations. Each is one node with a single analytic backward
    //     rule instead of being assembled from many smaller nodes, which is
    //     both faster and far lighter on memory. These are general array
    //     operations, so they live here. RMS normalization and the
    //     cross-entropy loss are also fused, but each belongs to one component
    //     (the norm layer, the trainer), so they are defined there using the
    //     same _inputs/_backward protocol.

    softMaxedRows() {
        const rows = this.rows;
        const cols = this.columns;
        const out = new Tensor(rows, cols);

        for (let row = 0; row < rows; ++row) {
            const base = row * cols;

            let rowMax = this.data[base];
            for (let col = 1; col < cols; ++col) {
                const v = this.data[base + col];
                if (v > rowMax) rowMax = v;
            }

            let rowSum = 0.0;
            for (let col = 0; col < cols; ++col) {
                const e = Math.exp(this.data[base + col] - rowMax);
                out.data[base + col] = e;
                rowSum += e;
            }
            for (let col = 0; col < cols; ++col) {
                out.data[base + col] /= rowSum;
            }
        }

        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                // For each row: dx_j = p_j * (g_j - sum_k g_k p_k)
                for (let row = 0; row < rows; ++row) {
                    const base = row * cols;
                    let dot = 0.0;
                    for (let col = 0; col < cols; ++col) {
                        dot += out.grad[base + col] * out.data[base + col];
                    }
                    for (let col = 0; col < cols; ++col) {
                        const p = out.data[base + col];
                        this.grad[base + col] += p * (out.grad[base + col] - dot);
                    }
                }
            };
        }
        return out;
    }

    causalMasked() {
        const rows = this.rows;
        const cols = this.columns;
        const out = new Tensor(rows, cols);

        for (let row = 0; row < rows; ++row) {
            for (let col = 0; col < cols; ++col) {
                const index = row * cols + col;
                out.data[index] = col > row ? -Infinity : this.data[index];
            }
        }

        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                // Masked cells are constants, so gradient only flows where col <= row.
                for (let row = 0; row < rows; ++row) {
                    for (let col = 0; col <= row; ++col) {
                        const index = row * cols + col;
                        this.grad[index] += out.grad[index];
                    }
                }
            };
        }
        return out;
    }

    // Embedding lookup: copy the rows named by `indices` into a new tensor.
    // The gradient scatters back, accumulating when an index repeats.
    gatherRows(indices) {
        const out = new Tensor(indices.length, this.columns);
        for (let r = 0; r < indices.length; ++r) {
            const src = indices[r] * this.columns;
            const dst = r * this.columns;
            for (let col = 0; col < this.columns; ++col) {
                out.data[dst + col] = this.data[src + col];
            }
        }
        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                for (let r = 0; r < indices.length; ++r) {
                    const src = indices[r] * this.columns;
                    const dst = r * this.columns;
                    for (let col = 0; col < this.columns; ++col) {
                        this.grad[src + col] += out.grad[dst + col];
                    }
                }
            };
        }
        return out;
    }

    // RoPE: rotate each row's component pairs by an angle set by the row's
    // position. Query and key rows sit in sequence order, so the row index is
    // the position. The angles are fixed functions of position and frequency,
    // not learned, so this op adds nothing to parameters(); the gradient only
    // routes back to the input. The cosine and sine of each angle are cached,
    // since the backward pass — a rotation by the negated angle — reuses them.
    ropeRotated(thetaBase = 10000) {
        const rows = this.rows;
        const cols = this.columns; // head_dim, must be even
        const halfCols = cols / 2;
        const out = new Tensor(rows, cols);

        // Frequency depends on the pair, not the row
        // Compute it once and cache
        const freqs = new Float32Array(halfCols);
        for (let pair = 0; pair < halfCols; ++pair) {
            freqs[pair] = 1.0 / Math.pow(thetaBase, (2 * pair) / cols);
        }

        // Cached for the backward pass.
        const cosTable = new Float32Array(rows * halfCols);
        const sinTable = new Float32Array(rows * halfCols);

        for (let row = 0; row < rows; ++row) {
            const base = row * cols;
            const trig = row * halfCols;
            for (let pair = 0; pair < halfCols; ++pair) {
                const angle = row * freqs[pair];
                const cos = Math.cos(angle);
                const sin = Math.sin(angle);
                cosTable[trig + pair] = cos;
                sinTable[trig + pair] = sin;

                const a = this.data[base + 2 * pair];
                const b = this.data[base + 2 * pair + 1];
                out.data[base + 2 * pair]     = a * cos - b * sin;
                out.data[base + 2 * pair + 1] = a * sin + b * cos;
            }
        }

        if (Tensor.gradEnabled) {
            out._inputs = [this];
            out._backward = () => {
                // Rotation is linear, the gradient is the inverse rotation.
                for (let row = 0; row < rows; ++row) {
                    const base = row * cols;
                    const trig = row * halfCols;
                    for (let pair = 0; pair < halfCols; ++pair) {
                        const cos = cosTable[trig + pair];
                        const sin = sinTable[trig + pair];
                        const ga = out.grad[base + 2 * pair];
                        const gb = out.grad[base + 2 * pair + 1];
                        this.grad[base + 2 * pair]     +=  ga * cos + gb * sin;
                        this.grad[base + 2 * pair + 1] += -ga * sin + gb * cos;
                    }
                }
            };
        }
        return out;
    }

    backward() {
        // Topological sort: inputs come before the nodes built from them.
        const topo = [];
        const visited = new Set();

        const visit = (node) => {
            if (visited.has(node)) return;
            visited.add(node);
            for (let i = 0; i < node._inputs.length; ++i) {
                visit(node._inputs[i]);
            }
            topo.push(node);
        };
        visit(this);

        // Make sure every node in the graph has a gradient buffer. Freshly
        // built tensors start at null and get a zeroed buffer here. Parameter
        // tensors keep the buffer the trainer already zeroed.
        for (let i = 0; i < topo.length; ++i) {
            if (topo[i].grad === null) {
                topo[i].grad = new Float32Array(topo[i].data.length);
            }
        }

        // Seed: derivative of the output with respect to itself is 1.
        this.grad.fill(1);

        // Walk in reverse. Each node pushes gradient onto its inputs.
        for (let i = topo.length - 1; i >= 0; --i) {
            topo[i]._backward();
        }
    }
}
