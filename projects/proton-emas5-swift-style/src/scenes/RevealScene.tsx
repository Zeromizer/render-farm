import React from 'react';
import {AbsoluteFill, Easing, Img, Interactive, interpolate, Sequence, staticFile, useCurrentFrame} from 'remotion';
import {RacingBackground} from '../components/RacingBackground';
import {DrivingCar} from '../components/DrivingCar';

export const RevealScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <RacingBackground darkRoad={frame >= 112} roadOpacity={interpolate(frame, [112, 124], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})} />
      <Interactive.Div name="All-new model title" style={{position: 'absolute', top: 300, left: 65, right: 65, color: 'white', textAlign: 'center', fontFamily: 'Arial Black, sans-serif', fontStyle: 'italic', fontWeight: 900, opacity: interpolate(frame, [0, 10, 80, 96], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [0, 22], ['0px 90px', '0px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)})}}>
        <div style={{fontSize: 54, letterSpacing: 4}}>THE ALL-NEW</div>
        <div style={{fontFamily: 'Impact, Arial Black, sans-serif', fontSize: 116, lineHeight: 0.95}}>PROTON</div>
        <div style={{fontFamily: 'Impact, Arial Black, sans-serif', fontSize: 168, lineHeight: 0.9}}>e.MAS 5</div>
      </Interactive.Div>
      <div style={{position: 'absolute', left: 180, right: 180, bottom: 300, height: 65, borderRadius: '50%', backgroundColor: 'rgba(51,0,5,0.45)', filter: 'blur(25px)', opacity: interpolate(frame, [5, 24, 82, 98], [0, 0.8, 0.8, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), scale: interpolate(frame, [5, 75], [0.55, 1.08], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', output: 'perceptual-scale'})}} />
      <Img name="Front reveal" src={staticFile('emas5/front-v8.png')} style={{position: 'absolute', width: 810, height: 'auto', left: '50%', bottom: 280, translate: interpolate(frame, [0, 24, 82, 102], ['-50% 720px', '-50% 0px', '-50% -10px', '-50% 1120px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)}), scale: interpolate(frame, [0, 82], [0.68, 1.05], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1), output: 'perceptual-scale'}), filter: 'drop-shadow(0 30px 24px rgba(0,0,0,0.42)) brightness(1.025)'}} />
      <Interactive.Div name="Feature promise" style={{position: 'absolute', bottom: 140, left: 90, right: 90, textAlign: 'center', color: 'white', fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: 72, lineHeight: 0.95, opacity: interpolate(frame, [44, 58, 82, 98], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [44, 62], ['0px 90px', '0px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)})}}>
        WITH EV INNOVATION<br />BUILT FOR EVERY DAY
      </Interactive.Div>
      <Sequence name="Side traverse and curved-road drive" from={86} durationInFrames={64} premountFor={24}>
        <DrivingCar />
      </Sequence>
    </AbsoluteFill>
  );
};
