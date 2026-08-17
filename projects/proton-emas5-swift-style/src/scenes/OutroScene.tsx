import React from 'react';
import {AbsoluteFill, Easing, Img, Interactive, interpolate, staticFile, useCurrentFrame} from 'remotion';

export const OutroScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{background: 'radial-gradient(circle at 50% 48%, #12a3da 0%, #096aa8 36%, #081d62 100%)', overflow: 'hidden'}}>
      <Interactive.Div name="Blue light sweep" style={{position: 'absolute', width: 1500, height: 360, left: -230, top: 780, borderRadius: 200, backgroundColor: '#76d7ff', opacity: 0.18, rotate: '-24deg', translate: interpolate(frame, [0, 194], ['-350px 0px', '380px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)})}} />
      <Interactive.Div name="Book a test drive call to action" style={{position: 'absolute', top: 280, left: 70, right: 70, zIndex: 2, color: 'white', textAlign: 'center', fontFamily: 'Arial Black, sans-serif', fontStyle: 'italic', textShadow: '0 7px 24px rgba(0,23,73,0.55)', opacity: interpolate(frame, [6, 22, 178, 194], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [6, 24], ['0px 55px', '0px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)})}}>
        <div style={{fontSize: 36, letterSpacing: 5}}>DISCOVER THE PROTON e.MAS 5</div>
        <div style={{fontSize: 76, lineHeight: 1.02, marginTop: 16}}>BOOK A TEST DRIVE TODAY</div>
        <div style={{width: 190, height: 8, margin: '28px auto 0', borderRadius: 999, background: 'linear-gradient(90deg, #ff562f, #ffb52f)'}} />
      </Interactive.Div>
      <Interactive.Div name="Official Proton e.MAS, Evolve Cars and contact details" style={{position: 'absolute', inset: 0, opacity: interpolate(frame, [12, 32, 178, 194], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), scale: interpolate(frame, [12, 40], [0.96, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.spring({damping: 180}), output: 'perceptual-scale'})}}>
        <Img name="Preserved Evolve Cars and dealer contact details" src={staticFile('ending/dealer-details-v9.png')} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain'}} />
        <Img name="Official Proton e.MAS dealer lockup" src={staticFile('ending/proton-emas-official-white-v9.png')} style={{position: 'absolute', top: 740, left: 145, width: 520, height: 'auto'}} />
      </Interactive.Div>
    </AbsoluteFill>
  );
};
