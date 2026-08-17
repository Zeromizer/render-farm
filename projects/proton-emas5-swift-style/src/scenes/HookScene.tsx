import React from 'react';
import {AbsoluteFill, Easing, Interactive, interpolate, useCurrentFrame} from 'remotion';
import {RacingBackground} from '../components/RacingBackground';

export const HookScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <RacingBackground />
      <Interactive.Div name="Hook copy" style={{position: 'absolute', top: 625, left: 110, right: 110, textAlign: 'center', color: 'white', fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontWeight: 900, lineHeight: 0.88, textShadow: '0 8px 0 rgba(89,0,8,0.35)', translate: interpolate(frame, [8, 26], ['0px 120px', '0px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)}), opacity: interpolate(frame, [7, 15], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>
        <div style={{fontSize: 122}}>ELECTRIC</div>
        <div style={{fontFamily: 'Arial Black, sans-serif', fontSize: 58, fontStyle: 'normal', lineHeight: 1.15}}>AND</div>
        <div style={{fontSize: 122, WebkitTextStroke: '3px white', color: 'transparent'}}>PRACTICAL</div>
        <div style={{fontFamily: 'Arial Black, sans-serif', fontSize: 57, fontStyle: 'normal', lineHeight: 1.3}}>IS NOW</div>
        <div style={{fontSize: 72, lineHeight: 1}}>AFFORDABLE TOO</div>
      </Interactive.Div>
    </AbsoluteFill>
  );
};
