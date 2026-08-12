import React from 'react';
import {AbsoluteFill, Easing, Img, Interactive, interpolate, Sequence, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {Video} from '@remotion/media';

type Caption = {text: string; startMs: number; endMs: number; timestampMs: number | null; confidence: number | null};
export const COLORS = {ink: '#071114', mint: '#6FF4D5', blue: '#77D9FF', orange: '#FF5635'};

export const FullVideo: React.FC<{src: string; name: string; trimBefore?: number; objectPosition?: string}> = ({src, name, trimBefore = 0, objectPosition = '50% 50%'}) => <Video name={name} src={staticFile(src)} trimBefore={trimBefore} muted style={{width: '100%', height: '100%', objectFit: 'cover', objectPosition}} />;
export const Shade: React.FC<{amount?: number}> = ({amount = .48}) => <AbsoluteFill style={{background: `linear-gradient(180deg, rgba(2,7,9,${amount * .55}) 0%, rgba(2,7,9,.12) 36%, rgba(2,7,9,${amount}) 100%)`}} />;

export const Eyebrow: React.FC<{children: React.ReactNode; color?: string}> = ({children, color = COLORS.mint}) => {
  const frame = useCurrentFrame();
  return <Interactive.Div name="Eyebrow" style={{fontFamily: 'Arial, Helvetica, sans-serif', fontSize: 30, fontWeight: 800, letterSpacing: 6, textTransform: 'uppercase', color, opacity: interpolate(frame, [3, 13], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [3, 13], ['0px 20px', '0px 0px'], {easing: Easing.bezier(.16,1,.3,1), extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>{children}</Interactive.Div>;
};

export const Headline: React.FC<{children: React.ReactNode; size?: number}> = ({children, size = 112}) => {
  const frame = useCurrentFrame();
  return <Interactive.Div name="Headline" style={{fontFamily: 'Arial, Helvetica, sans-serif', fontSize: size, lineHeight: .92, fontWeight: 900, letterSpacing: -5, color: 'white', marginTop: 20, opacity: interpolate(frame, [7, 20], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [7, 20], ['0px 50px', '0px 0px'], {easing: Easing.bezier(.16,1,.3,1), extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>{children}</Interactive.Div>;
};

export const SpecCard: React.FC<{value: string; label: string; x: number; y: number; color?: string; delay?: number}> = ({value, label, x, y, color = COLORS.mint, delay = 0}) => {
  const frame = useCurrentFrame();
  return <Interactive.Div name={`${label} specification`} style={{position: 'absolute', left: x, top: y, minWidth: 300, padding: '20px 24px 18px', borderLeft: `5px solid ${color}`, borderRadius: 8, backgroundColor: 'rgba(5,14,18,.72)', boxShadow: '0 20px 60px rgba(0,0,0,.32)', backdropFilter: 'blur(8px)', opacity: interpolate(frame, [delay + 5, delay + 16], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [delay + 5, delay + 16], ['0px 28px', '0px 0px'], {easing: Easing.bezier(.16,1,.3,1), extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}><div style={{fontFamily: 'Arial, Helvetica, sans-serif', fontSize: 58, lineHeight: 1, fontWeight: 900, color: 'white'}}>{value}</div><div style={{fontFamily: 'Arial, Helvetica, sans-serif', fontSize: 24, lineHeight: 1.2, fontWeight: 700, letterSpacing: 2, color, marginTop: 10, textTransform: 'uppercase'}}>{label}</div></Interactive.Div>;
};

export const captions: Caption[] = [
  {text: 'Buying your first car?', startMs: 200, endMs: 1700, timestampMs: null, confidence: null},
  {text: 'Before the usual choices… look at this.', startMs: 1700, endMs: 4000, timestampMs: null, confidence: null},
  {text: 'The e.MAS 5: compact, modern, city-ready.', startMs: 4100, endMs: 7000, timestampMs: null, confidence: null},
  {text: 'Need more room? The e.MAS 7 adds space and comfort.', startMs: 7000, endMs: 10000, timestampMs: null, confidence: null},
  {text: 'EV efficiency. Modern tech. Everyday practicality.', startMs: 10100, endMs: 15300, timestampMs: null, confidence: null},
  {text: 'Without jumping straight to the premium end.', startMs: 15300, endMs: 21000, timestampMs: null, confidence: null},
  {text: 'Compact and easy—or spacious enough to grow.', startMs: 21100, endMs: 26000, timestampMs: null, confidence: null},
  {text: 'Could the Proton e.MAS be your first car?', startMs: 26200, endMs: 29800, timestampMs: null, confidence: null}
];

export const CaptionLayer: React.FC = () => {
  const {fps} = useVideoConfig();
  return <AbsoluteFill style={{pointerEvents: 'none', zIndex: 10}}>{captions.map((caption, index) => <Sequence key={caption.startMs} from={Math.round(caption.startMs / 1000 * fps)} durationInFrames={Math.round((caption.endMs - caption.startMs) / 1000 * fps)}><CaptionPill text={caption.text} index={index} /></Sequence>)}</AbsoluteFill>;
};

const CaptionPill: React.FC<{text: string; index: number}> = ({text, index}) => {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{justifyContent: 'flex-end', alignItems: 'center', padding: '0 78px 235px'}}><Interactive.Div name={`Subtitle ${index + 1}`} style={{maxWidth: 900, padding: '15px 24px 17px', borderRadius: 16, backgroundColor: 'rgba(2,7,9,.76)', color: 'white', fontFamily: 'Arial, Helvetica, sans-serif', fontSize: 39, lineHeight: 1.18, fontWeight: 800, textAlign: 'center', boxShadow: '0 12px 38px rgba(0,0,0,.25)', opacity: interpolate(frame, [0, 5], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [0, 5], ['0px 18px', '0px 0px'], {easing: Easing.bezier(.16,1,.3,1), extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>{text}</Interactive.Div></AbsoluteFill>;
};
