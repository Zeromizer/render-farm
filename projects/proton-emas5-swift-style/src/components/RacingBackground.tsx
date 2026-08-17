import React from 'react';
import {AbsoluteFill, Easing, Interactive, interpolate, useCurrentFrame} from 'remotion';
import {MovingRoad} from './MovingRoad';

export const RacingBackground: React.FC<{darkRoad?: boolean; roadOpacity?: number}> = ({darkRoad = false, roadOpacity = 1}) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{backgroundColor: '#00AEEF', overflow: 'hidden'}}>
      <Interactive.Div name="Sweeping blue ribbon" style={{position: 'absolute', width: 1450, height: 470, top: 110, left: -230, backgroundColor: '#007EAE', opacity: 0.34, rotate: '-24deg', borderRadius: 180, translate: interpolate(frame, [0, 120], ['-70px 0px', '95px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)})}} />
      <Interactive.Div name="Electric blue highlight ribbon" style={{position: 'absolute', width: 1350, height: 230, top: 515, left: -110, backgroundColor: '#6AD8FF', opacity: 0.26, rotate: '-24deg', borderRadius: 150, translate: interpolate(frame, [0, 120], ['80px 0px', '-85px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}} />
      {darkRoad ? <MovingRoad opacity={roadOpacity} /> : null}
    </AbsoluteFill>
  );
};
