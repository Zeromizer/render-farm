import React from 'react';
import {AbsoluteFill, Easing, Img, interpolate, staticFile, useCurrentFrame} from 'remotion';

const SideProfile: React.FC = () => {
  const frame = useCurrentFrame();
  const wheelRotation = interpolate(frame, [0, 38], ['0deg', '650deg'], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.bezier(0.28, 0.05, 0.22, 1),
  });
  const opacity = interpolate(frame, [0, 4, 30, 38], [0, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const translateX = interpolate(frame, [0, 38], [-980, 185], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.bezier(0.28, 0.05, 0.22, 1),
  });
  const speedBlur = interpolate(frame, [0, 8, 27, 38], [13, 3, 3, 15], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const suspension = Math.sin(frame * 0.72) * 3.5;

  return (
    <div
      style={{
        position: 'absolute',
        left: 0,
        top: 720,
        width: 1120,
        height: 458,
        opacity,
        translate: `${translateX}px ${suspension}px`,
        scale: interpolate(frame, [0, 38], [0.84, 1.035], {
          extrapolateLeft: 'clamp',
          extrapolateRight: 'clamp',
          easing: Easing.bezier(0.28, 0.05, 0.22, 1),
          output: 'perceptual-scale',
        }),
        rotate: interpolate(frame, [0, 26, 38], ['-1.6deg', '0deg', '0.7deg'], {
          extrapolateLeft: 'clamp',
          extrapolateRight: 'clamp',
        }),
      }}
    >
      <div
        style={{
          position: 'absolute',
          left: 122,
          top: 398,
          width: 905,
          height: 46,
          borderRadius: '50%',
          backgroundColor: 'rgba(15,8,7,0.58)',
          filter: 'blur(18px)',
          scale: interpolate(frame, [0, 38], [0.78, 1.08], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
            output: 'perceptual-scale',
          }),
        }}
      />
      <Img
        name="Side-driving motion trail"
        src={staticFile('emas5/side.png')}
        style={{
          position: 'absolute',
          width: 1120,
          height: 'auto',
          left: -42,
          top: 1,
          opacity: interpolate(frame, [0, 7, 27, 38], [0.24, 0.05, 0.05, 0.28], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          }),
          filter: `blur(${speedBlur}px) brightness(1.08)`,
          scale: '1.035 1',
        }}
      />
      <Img
        name="Proton e.MAS 5 side profile"
        src={staticFile('emas5/side.png')}
        style={{position: 'absolute', width: 1120, height: 'auto', left: 0, top: 0, filter: 'drop-shadow(0 15px 12px rgba(0,0,0,0.24))'}}
      />
      <Img
        name="Rotating rear alloy"
        src={staticFile('emas5/wheels/rear-rim.png')}
        style={{position: 'absolute', width: 123, height: 123, left: 129, top: 306, rotate: wheelRotation}}
      />
      <Img
        name="Rotating front alloy"
        src={staticFile('emas5/wheels/front-rim.png')}
        style={{position: 'absolute', width: 120, height: 120, left: 885, top: 311, rotate: wheelRotation}}
      />
      <div
        style={{
          position: 'absolute',
          left: 260,
          top: 28,
          width: 660,
          height: 210,
          borderRadius: '50%',
          background: 'linear-gradient(105deg, transparent 5%, rgba(255,239,206,0.2) 50%, transparent 78%)',
          mixBlendMode: 'screen',
          translate: interpolate(frame, [0, 38], ['-180px 0px', '260px 0px'], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          }),
          rotate: '-8deg',
          filter: 'blur(18px)',
        }}
      />
    </div>
  );
};
export const DrivingCar: React.FC = () => {
  const frame = useCurrentFrame();
  const curveOpacity = interpolate(frame, [31, 40, 61, 64], [0, 1, 1, 0.92], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill>
      <SideProfile />
      <div
        style={{
          position: 'absolute',
          left: 76,
          bottom: 215,
          width: 850,
          height: 72,
          borderRadius: '50%',
          backgroundColor: 'rgba(7,8,12,0.6)',
          filter: 'blur(24px)',
          opacity: curveOpacity * 0.9,
          scale: interpolate(frame, [31, 48, 64], [0.56, 1, 1.06], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
            easing: Easing.bezier(0.16, 1, 0.3, 1),
            output: 'perceptual-scale',
          }),
        }}
      />
      <Img
        name="Curved-road Proton e.MAS 5"
        src={staticFile('emas5/front-left.png')}
        style={{
          position: 'absolute',
          width: 1030,
          height: 'auto',
          left: '50%',
          bottom: 185,
          opacity: curveOpacity,
          translate: interpolate(frame, [31, 48, 64], ['-940px 105px', '-515px 0px', '-475px -18px'], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
          scale: interpolate(frame, [31, 48, 64], [0.72, 0.97, 1.035], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
            easing: Easing.bezier(0.16, 1, 0.3, 1),
            output: 'perceptual-scale',
          }),
          rotate: interpolate(frame, [31, 48, 64], ['-4deg', '-0.5deg', '0.8deg'], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          }),
          filter: 'drop-shadow(0 28px 22px rgba(0,0,0,0.48)) brightness(1.035)',
        }}
      />
      <AbsoluteFill
        style={{
          background: 'radial-gradient(circle at 70% 58%, rgba(255,238,177,0.96), rgba(255,79,55,0.52) 34%, transparent 66%)',
          opacity: interpolate(frame, [26, 31, 36, 43], [0, 0.95, 0.54, 0], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          }),
          mixBlendMode: 'screen',
        }}
      />
    </AbsoluteFill>
  );
};
