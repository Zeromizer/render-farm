import React from 'react';
import {AbsoluteFill, Easing, Img, Interactive, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {RacingBackground} from '../components/RacingBackground';

export const OfferScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <RacingBackground />
      <Img name="Final front hero" src={staticFile('emas5/front.png')} style={{position: 'absolute', width: 850, height: 'auto', left: '50%', top: 360, translate: '-50% 0px', scale: interpolate(frame, [0, 32], [0.72, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1), output: 'perceptual-scale'}), filter: 'drop-shadow(0 32px 25px rgba(0,0,0,0.48))'}} />
      <Interactive.Div name="Final model name" style={{position: 'absolute', top: 230, left: 70, right: 70, color: 'white', textAlign: 'center', fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: 120, opacity: interpolate(frame, [0, 9], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [0, 12], ['0px -70px', '0px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>e.MAS 5</Interactive.Div>
      <Interactive.Div name="Range callout" style={{position: 'absolute', bottom: 165, left: 70, right: 70, color: 'white', textAlign: 'center', fontFamily: 'Arial Black, sans-serif', fontStyle: 'italic', opacity: interpolate(frame, [14, 24], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>
        <div style={{fontSize: 48}}>UP TO</div>
        <div style={{fontFamily: 'Impact, Arial Black, sans-serif', fontSize: 154, lineHeight: 0.9}}>325 KM</div>
        <div style={{fontSize: 48, marginTop: 12}}>WLTP RANGE*</div>
      </Interactive.Div>
      <Interactive.Div name="Range disclaimer" style={{position: 'absolute', bottom: 54, left: 70, right: 70, color: 'white', textAlign: 'right', fontFamily: 'Arial, sans-serif', fontSize: 20, opacity: 0.85}}>*Range varies by variant, driving style and conditions.</Interactive.Div>
    </AbsoluteFill>
  );
};
