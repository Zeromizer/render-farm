import React from 'react';
import {AbsoluteFill, Easing, Interactive, interpolate, useCurrentFrame} from 'remotion';
import {COLORS, Eyebrow, FullVideo, Headline, Shade} from '../components';

export const HookScene: React.FC = () => {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{backgroundColor: COLORS.ink}}><FullVideo name="Opening rolling e.MAS 7" src="emas7/front-reveal.mp4" trimBefore={8} objectPosition="48% 50%" /><Shade amount={.72} /><AbsoluteFill style={{padding: '145px 78px 0', fontFamily: 'Arial, Helvetica, sans-serif'}}><Eyebrow>Singapore · your first EV</Eyebrow><Headline size={128}>Why the Proton<br /><span style={{color: COLORS.mint}}>e.MAS</span> should be<br />your first car.</Headline></AbsoluteFill><Interactive.Div name="Hook accent line" style={{position: 'absolute', left: 78, top: 690, width: interpolate(frame, [18, 32], [0, 450], {easing: Easing.bezier(.16,1,.3,1), extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), height: 10, borderRadius: 999, backgroundColor: COLORS.mint}} /></AbsoluteFill>;
};
