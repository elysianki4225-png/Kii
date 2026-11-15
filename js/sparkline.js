export class Sparkline {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.maxPoints = options.maxPoints || 48;
    this.strokeColor = options.strokeColor || '#2F6FED';
    this.fillTop = options.fillTop || 'rgba(47, 111, 237, 0.18)';
    this.fillBottom = options.fillBottom || 'rgba(47, 111, 237, 0.0)';
    this.points = [];
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this._resize();
  }

  _resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(120, Math.round(rect.width || 300));
    const h = Math.max(44, Math.round(rect.height || 56));
    const targetW = Math.round(w * this.dpr);
    const targetH = Math.round(h * this.dpr);
    if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
      this.canvas.width = targetW;
      this.canvas.height = targetH;
    }
    this.width = w;
    this.height = h;
  }

  push(value) {
    const v = Number.isFinite(value) && value >= 0 ? value : 0;
    this.points.push(v);
    if (this.points.length > this.maxPoints) {
      this.points.shift();
    }
    this.draw();
  }

  reset() {
    this.points = [];
    this.draw();
  }

  draw() {
    if (!this.ctx) return;
    this._resize();
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    ctx.save();
    ctx.scale(this.dpr, this.dpr);
    ctx.clearRect(0, 0, w, h);

    // Subtle horizontal guide lines
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.13)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    for (let i = 1; i <= 2; i++) {
      const y = Math.round((h / 3) * i) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    const n = this.points.length;
    if (n < 2) {
      // Idle flat baseline
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(6, h - 10);
      ctx.lineTo(w - 6, h - 10);
      ctx.stroke();
      ctx.restore();
      return;
    }

    let maxVal = 0;
    for (let i = 0; i < n; i++) {
      if (this.points[i] > maxVal) maxVal = this.points[i];
    }
    if (maxVal <= 0) maxVal = 1;
    // Headroom so curve never clips top edge
    const ceiling = maxVal * 1.15;
    const padX = 4;
    const padTop = 6;
    const padBottom = 6;
    const plotW = w - padX * 2;
    const plotH = h - padTop - padBottom;
    const stepX = plotW / (this.maxPoints - 1);
    const startIdx = this.maxPoints - n;

    const coords = new Array(n);
    for (let i = 0; i < n; i++) {
      const x = padX + (startIdx + i) * stepX;
      const ratio = Math.min(1, this.points[i] / ceiling);
      const y = padTop + plotH * (1 - ratio);
      coords[i] = { x, y };
    }

    // Area gradient fill
    const grad = ctx.createLinearGradient(0, padTop, 0, h - padBottom);
    grad.addColorStop(0, this.fillTop);
    grad.addColorStop(1, this.fillBottom);

    ctx.beginPath();
    ctx.moveTo(coords[0].x, h - padBottom);
    ctx.lineTo(coords[0].x, coords[0].y);
    for (let i = 1; i < n; i++) {
      const prev = coords[i - 1];
      const curr = coords[i];
      const midX = (prev.x + curr.x) * 0.5;
      ctx.quadraticCurveTo(prev.x, prev.y, midX, (prev.y + curr.y) * 0.5);
    }
    const last = coords[n - 1];
    ctx.lineTo(last.x, last.y);
    ctx.lineTo(last.x, h - padBottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Crisp stroke line
    ctx.beginPath();
    ctx.moveTo(coords[0].x, coords[0].y);
    for (let i = 1; i < n; i++) {
      const prev = coords[i - 1];
      const curr = coords[i];
      const midX = (prev.x + curr.x) * 0.5;
      ctx.quadraticCurveTo(prev.x, prev.y, midX, (prev.y + curr.y) * 0.5);
    }
    ctx.lineTo(last.x, last.y);
    ctx.strokeStyle = this.strokeColor;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Live endpoint dot
    ctx.beginPath();
    ctx.arc(last.x, last.y, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = this.strokeColor;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#F4F6FA';
    ctx.stroke();

    ctx.restore();
  }
}
