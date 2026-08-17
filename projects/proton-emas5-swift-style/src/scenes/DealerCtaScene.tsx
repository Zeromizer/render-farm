import React from 'react';
import {Video} from '@remotion/media';
import {AbsoluteFill, Easing, Interactive, interpolate, staticFile, useCurrentFrame} from 'remotion';

export const DealerCtaScene: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{backgroundColor: '#000', overflow: 'hidden'}}>
      <Video
        name="Correct Proton e.MAS dealer end card"
        src={staticFile('ending/proton-logo-correct.mp4')}
        durationInFrames={150}
        muted
        style={{position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover'}}
      />
      <Interactive.Div
        name="Book a test drive call to action"
        style={{position: 'absolute', top: 360, left: 70, right: 70, color: 'white', textAlign: 'center', fontFamily: 'Arial Black, sans-serif', fontStyle: 'italic', textShadow: '0 6px 22px rgba(0,0,0,0.8)', opacity: interpolate(frame, [8, 20, 126, 143], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [8, 24], ['0px 48px', '0px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)})}}
      >
        <div style={{fontSize: 34, letterSpacing: 5}}>DISCOVER THE PROTON e.MAS 5</div>
        <div style={{fontSize: 70, lineHeight: 1.05, marginTop: 16}}>BOOK A TEST DRIVE TODAY</div>
        <div style={{width: 190, height: 8, margin: '26px auto 0', borderRadius: 999, background: 'linear-gradient(90deg, #d6001c, #ff5d2d)'}} />
      </Interactive.Div>
    </AbsoluteFill>
  );
};
