/**
 * Web Audio API 音效模块
 * 提供分时页面买入/卖出信号音效播放，支持冷却时间和音量控制
 */

let audioContext: AudioContext | null = null;
let lastPlayed: { buy: number; sell: number } = { buy: 0, sell: 0 };

function getAudioContext(): AudioContext | null {
  if (!audioContext) {
    try {
      audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    } catch {
      return null;
    }
  }
  if (audioContext.state === 'suspended') {
    audioContext.resume().catch(() => {});
  }
  return audioContext;
}

/**
 * 播放买入/卖出信号音效
 * @param type - 信号类型 'buy' | 'sell'
 * @param cooldownSeconds - 同一类型冷却时间（秒），默认 30
 * @param volume - 音量 0.0~1.0，默认 0.3
 */
export function playSignalSound(
  type: 'buy' | 'sell',
  cooldownSeconds: number = 30,
  volume: number = 0.3,
): void {
  const now = Date.now();
  if (now - lastPlayed[type] < cooldownSeconds * 1000) return;
  lastPlayed[type] = now;

  const ctx = getAudioContext();
  if (!ctx) return;

  // 买入：上行音阶 C5→E5→G5（523→659→784 Hz），每音 300ms，总时长约 1s
  // 卖出：下行音阶 G5→E5→C5（784→659→523 Hz），每音 300ms，总时长约 1s
  const notes = type === 'buy'
    ? [523.25, 659.25, 783.99]  // C5, E5, G5
    : [783.99, 659.25, 523.25]; // G5, E5, C5
  const noteDuration = 0.3; // 300ms per note

  notes.forEach((freq, i) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(volume, ctx.currentTime + i * noteDuration);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + (i + 1) * noteDuration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(ctx.currentTime + i * noteDuration);
    osc.stop(ctx.currentTime + (i + 1) * noteDuration + 0.05);
  });
}