import React from 'react';
import {AbsoluteFill, Easing, Img, Interactive, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {RacingBackground} from '../components/RacingBackground';

export const RevealScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <RacingBackground darkRoad={frame >= 88} />
      <Interactive.Div name="All-new model title" style={{position: 'absolute', top: 300, left: 65, right: 65, color: 'white', textAlign: 'center', fontFamily: 'Arial Black, sans-serif', fontStyle: 'italic', fontWeight: 900, opacity: interpolate(frame, [0, 8, 76, 90], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [0, 18], ['0px 90px', '0px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)})}}>
        <div style={{fontSize: 54, letterSpacing: 4}}>THE ALL-NEW</div>
        <div style={{fontFamily: 'Impact, Arial Black, sans-serif', fontSize: 116, lineHeight: 0.95}}>PROTON</div>
        <div style={{fontFamily: 'Impact, Arial Black, sans-serif', fontSize: 168, lineHeight: 0.9}}>e.MAS 5</div>
      </Interactive.Div>
      <Img name="Front reveal" src={staticFile('emas5/front.png')} style={{position: 'absolute', width: 810, height: 'auto', left: '50%', bottom: 280, translate: interpolate(frame, [0, 18, 76, 96], ['-50% 720px', '-50% 0px', '-50% 0px', '-50% 700px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)}), scale: interpolate(frame, [0, 75], [0.68, 1.05], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1), output: 'perceptual-scale'}), filter: 'drop-shadow(0 30px 24px rgba(0,0,0,0.42))'}} />
      <Interactive.Div name="Feature promise" style={{position: 'absolute', bottom: 140, left: 90, right: 90, textAlign: 'center', color: 'white', fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: 72, lineHeight: 0.95, opacity: interpolate(frame, [40, 50, 78, 88], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [40, 56], ['0px 90px', '0px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>
        WITH EV INNOVATION<br />BUILT FOR EVERY DAY
      </Interactive.Div>
      <Img name="Road-driving angle" src={staticFile('emas5/front-left.png')} style={{position: 'absolute', width: 1020, height: 'auto', left: '50%', bottom: 60, opacity: interpolate(frame, [86, 96], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [86, 119], ['-1150px 0px', '-459px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.12, 0.7, 0.2, 1)}), filter: 'drop-shadow(0 30px 24px rgba(0,0,0,0.45))'}} />
    </AbsoluteFill>
  );
};
