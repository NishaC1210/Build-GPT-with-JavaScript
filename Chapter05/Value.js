class Value {
    constructor(data, inputs = []) {
        this.data = data;
        this.partial = 0;
        this._inputs = inputs;
        this._backward = () => {}; // leaves do nothing
    }

    add(other) {
        if (!(other instanceof Value)) other = new Value(other);
        const out = new Value(this.data + other.data, [this, other]);
        out._backward = () => {
            // d(a + b)/da = 1, d(a + b)/db = 1
            this.partial  += out.partial * 1.0;
            other.partial += out.partial * 1.0;
        };
        return out;
    }

    mul(other) {
        if (!(other instanceof Value)) other = new Value(other);
        const out = new Value(this.data * other.data, [this, other]);
        out._backward = () => {
            // d(a * b)/da = b, d(a * b)/db = a
            this.partial  += other.data * out.partial;
            other.partial += this.data  * out.partial;
        };
        return out;
    }

    pow(exponent) {
        // exponent is a plain number, not a Value
        const out = new Value(Math.pow(this.data, exponent), [this]);
        out._backward = () => {
            // d(a^k)/da = k * a^(k-1)
            this.partial += exponent * Math.pow(this.data, exponent - 1) * out.partial;
        };
        return out;
    }

    relu() {
        const out = new Value(this.data > 0 ? this.data : 0, [this]);
        out._backward = () => {
            // d/dx max(0, x) = 1 if x > 0 else 0
            this.partial += (this.data > 0 ? 1 : 0) * out.partial;
        };
        return out;
    }

    exp() {
        const out = new Value(Math.exp(this.data), [this]);
        out._backward = () => {
            // d/dx exp(x) = exp(x), which is exactly out.data
            this.partial += out.data * out.partial;
        };
        return out;
    }

    log() {
        const out = new Value(Math.log(this.data), [this]);
        out._backward = () => {
            // d/dx log(x) = 1/x
            this.partial += (1 / this.data) * out.partial;
        };
        return out;
    }

    neg()      { return this.mul(-1); }
    sub(other) { return this.add((other instanceof Value ? other : new Value(other)).neg()); }
    div(other) { return this.mul((other instanceof Value ? other : new Value(other)).pow(-1)); }
    sqrt()     { return this.pow(0.5); }

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

        // Seed: derivative of output w.r.t. itself.
        this.partial = 1;

        // Walk in reverse: roots first, leaves last.
        // Each node pushes gradient onto its _inputs via the chain rule.
        for (let i = topo.length - 1; i >= 0; --i) {
            topo[i]._backward();
        }
    }
} 