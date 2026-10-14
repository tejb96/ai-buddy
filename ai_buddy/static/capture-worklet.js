// Microphone capture: low-pass + resample to 16 kHz mono PCM16, posted in 100 ms frames.
// Keeps a short pre-roll so the first syllable isn't lost when talk is pressed a beat late.

const OUT_RATE = 16000;
const FRAME = 1600;          // 100 ms at 16 kHz
const PREROLL = 4800;        // 300 ms
const TAPS = 63;

function lowpass(cutoff, rate, taps) {
  // Windowed-sinc FIR (Blackman), unity DC gain.
  const fc = cutoff / rate, m = taps - 1, h = new Float32Array(taps);
  let sum = 0;
  for (let i = 0; i < taps; i++) {
    const n = i - m / 2;
    const sinc = n === 0 ? 2 * Math.PI * fc : Math.sin(2 * Math.PI * fc * n) / n;
    const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / m) + 0.08 * Math.cos((4 * Math.PI * i) / m);
    h[i] = sinc * w;
    sum += h[i];
  }
  for (let i = 0; i < taps; i++) h[i] /= sum;
  return h;
}

class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / OUT_RATE;
    this.h = lowpass(Math.min(7200, sampleRate / 2 * 0.9), sampleRate, TAPS);
    this.ring = new Float32Array(TAPS);
    this.ringPos = 0;
    this.n = 0;              // input samples consumed
    this.nextT = 0;          // input-time of the next output sample
    this.prevY = 0;
    this.pre = new Int16Array(PREROLL);
    this.prePos = 0;
    this.preFull = false;
    this.out = new Int16Array(FRAME);
    this.outLen = 0;
    this.active = false;
    this.levelAcc = 0;
    this.levelN = 0;
    this.port.onmessage = (e) => {
      if (e.data === "start") {
        this.active = true;
        this.outLen = 0;
        this.flushPreroll();
      } else if (e.data === "stop") {
        if (this.outLen) this.post(this.out.slice(0, this.outLen));
        this.outLen = 0;
        this.active = false;
        this.port.postMessage({ type: "stopped" });
      }
    };
  }

  flushPreroll() {
    const n = this.preFull ? PREROLL : this.prePos;
    const start = this.preFull ? this.prePos : 0;
    for (let i = 0; i < n; i++) this.emit(this.pre[(start + i) % PREROLL]);
    this.prePos = 0;
    this.preFull = false;
  }

  post(int16) {
    this.port.postMessage({ type: "audio", pcm: int16.buffer }, [int16.buffer]);
  }

  emit(s) {
    this.out[this.outLen++] = s;
    if (this.outLen === FRAME) {
      this.post(this.out);
      this.out = new Int16Array(FRAME);
      this.outLen = 0;
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const h = this.h;
    for (let i = 0; i < ch.length; i++) {
      const x = ch[i];
      this.levelAcc += x * x;
      this.levelN++;
      this.ring[this.ringPos] = x;
      this.ringPos = (this.ringPos + 1) % TAPS;
      let y = 0;
      for (let k = 0, p = this.ringPos; k < TAPS; k++, p = (p + 1) % TAPS) y += h[k] * this.ring[p];
      while (this.nextT <= this.n) {
        const frac = this.nextT - (this.n - 1);
        const v = this.prevY + (y - this.prevY) * frac;
        const s = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
        if (this.active) this.emit(s);
        else {
          this.pre[this.prePos] = s;
          this.prePos = (this.prePos + 1) % PREROLL;
          if (this.prePos === 0) this.preFull = true;
        }
        this.nextT += this.ratio;
      }
      this.prevY = y;
      this.n++;
    }
    if (this.levelN >= sampleRate / 50) {   // every 20 ms
      const rms = Math.sqrt(this.levelAcc / this.levelN);
      this.port.postMessage({ type: "level", db: 20 * Math.log10(rms + 1e-9) });
      this.levelAcc = 0;
      this.levelN = 0;
    }
    return true;
  }
}

registerProcessor("capture", Capture);
