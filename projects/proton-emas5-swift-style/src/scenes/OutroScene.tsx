import React from 'react';
import {AbsoluteFill, Easing, Interactive, interpolate, useCurrentFrame} from 'remotion';

export const OutroScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{background: 'radial-gradient(circle at 50% 48%, #12a3da 0%, #096aa8 36%, #081d62 100%)', overflow: 'hidden'}}>
      <Interactive.Div name="Blue light sweep" style={{position: 'absolute', width: 1500, height: 360, left: -230, top: 780, borderRadius: 200, backgroundColor: '#76d7ff', opacity: 0.18, rotate: '-24deg', translate: interpolate(frame, [0, 59], ['-350px 0px', '380px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)})}} />
      <Interactive.Div name="Proton e.MAS lockup" style={{position: 'absolute', top: 820, left: 80, right: 80, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 24, color: 'white', fontFamily: 'Arial, sans-serif', fontSize: 50, letterSpacing: 3, fontWeight: 900, opacity: interpolate(frame, [3, 14], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), scale: interpolate(frame, [0, 18], [0.78, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.spring({damping: 180}), output: 'perceptual-scale'})}}>
        <div style={{width: 48, height: 48, border: '7px solid white', borderRadius: 11, rotate: '45deg'}} />
        <span>PROTON</span><span style={{fontWeight: 500}}>e.MAS</span>
      </Interactive.Div>
      <Interactive.Div name="Outro tagline" style={{position: 'absolute', top: 915, left: 120, right: 120, textAlign: 'center', color: 'white', fontFamily: 'Arial Black, sans-serif', fontSize: 28, letterSpacing: 8, opacity: interpolate(frame, [16, 28], [0, 0.9], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>ELECTRIFY YOUR LIFE</Interactive.Div>
    </AbsoluteFill>
  );
};
