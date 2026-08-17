import React from 'react';
import {AbsoluteFill, Easing, Interactive, interpolate, useCurrentFrame} from 'remotion';

export const RacingBackground: React.FC<{darkRoad?: boolean}> = ({darkRoad = false}) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{background: 'linear-gradient(145deg, #d10a20 0%, #9f0312 58%, #c2081b 100%)', overflow: 'hidden'}}>
      <Interactive.Div name="Sweeping red ribbon" style={{position: 'absolute', width: 1450, height: 470, top: 110, left: -230, backgroundColor: '#7e0713', opacity: 0.34, rotate: '-24deg', borderRadius: 180, translate: interpolate(frame, [0, 120], ['-70px 0px', '95px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)})}} />
      <Interactive.Div name="Highlight ribbon" style={{position: 'absolute', width: 1350, height: 230, top: 515, left: -110, backgroundColor: '#ef3141', opacity: 0.23, rotate: '-24deg', borderRadius: 150, translate: interpolate(frame, [0, 120], ['80px 0px', '-85px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}} />
      {darkRoad ? <Interactive.Div name="Charcoal road" style={{position: 'absolute', left: -110, bottom: -150, width: 1320, height: 820, backgroundColor: '#292a2e', borderTopLeftRadius: 520, borderTopRightRadius: 120, rotate: '-3deg'}} /> : null}
    </AbsoluteFill>
  );
};
