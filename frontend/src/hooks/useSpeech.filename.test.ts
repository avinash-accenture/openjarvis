import { describe, expect, it } from 'vitest';
import { recordingFilename } from './useSpeech';

describe('recording upload names', () => {
  it.each([
    ['audio/mp4', 'recording.m4a'],
    ['audio/mp4;codecs=mp4a.40.2', 'recording.m4a'],
    ['audio/aac', 'recording.m4a'],
    ['audio/webm;codecs=opus', 'recording.webm'],
    ['audio/ogg', 'recording.ogg'],
    ['audio/wav', 'recording.wav'],
    ['', 'recording.webm'],
  ])('%s -> %s', (mime, name) => {
    expect(recordingFilename(mime)).toBe(name);
  });
});
