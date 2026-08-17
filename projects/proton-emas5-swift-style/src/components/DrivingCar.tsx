import React from 'react';
import {Video} from '@remotion/media';
import {AbsoluteFill, Easing, interpolate, staticFile, useCurrentFrame} from 'remotion';

const SOURCE = 'footage/emas5-freedom-1080p.mp4';

export const DrivingCar: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill>
      <Video
        name="Ambient road footage"
        src={staticFile(SOURCE)}
        trimBefore={1770}
        durationInFrames={64}
        muted
        objectFit="cover"
        style={{
          position: 'absolute',
          inset: -70,
          width: 1220,
          height: 2060,
          objectFit: 'cover',
          objectPosition: '58% 50%',
          filter: 'blur(34px) saturate(0.72) brightness(0.52)',
          opacity: interpolate(frame, [0, 8, 54, 63], [0, 0.42, 0.42, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
          scale: 1.08,
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: 42,
          right: 42,
          top: 585,
          height: 790,
          borderRadius: 64,
          overflow: 'hidden',
          clipPath: 'polygon(0 9%, 100% 0, 100% 91%, 0 100%)',
          backgroundColor: '#12151a',
          boxShadow: '0 42px 80px rgba(35,0,4,0.58)',
          opacity: interpolate(frame, [0, 7, 57, 63], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
          scale: interpolate(frame, [0, 18, 63], [0.92, 1, 1.035], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1), output: 'perceptual-scale'}),
        }}
      >
        <Video
          name="Authentic front tracking shot"
          src={staticFile(SOURCE)}
          trimBefore={1770}
          durationInFrames={25}
          muted
          objectFit="cover"
          style={{position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', objectPosition: '58% 50%', scale: 1.43, filter: 'saturate(0.88) contrast(1.08) brightness(0.88)', opacity: interpolate(frame, [0, 5, 17, 24], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}
        />
        <Video
          name="Authentic rear movement shot"
          src={staticFile(SOURCE)}
          trimBefore={3960}
          from={18}
          durationInFrames={46}
          muted
          objectFit="cover"
          style={{position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', objectPosition: '50% 50%', scale: 1.43, filter: 'saturate(0.86) contrast(1.08) brightness(0.84)', opacity: interpolate(frame, [18, 25, 58, 63], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}
        />
        <AbsoluteFill style={{background: 'linear-gradient(180deg, rgba(151,5,21,0.22), transparent 35%, transparent 66%, rgba(24,0,4,0.62))', mixBlendMode: 'multiply'}} />
        <AbsoluteFill style={{background: 'linear-gradient(108deg, rgba(255,255,255,0.2), transparent 22%, transparent 74%, rgba(255,48,66,0.22))', mixBlendMode: 'screen'}} />
      </div>
      <div
        style={{
          position: 'absolute',
          left: 92,
          right: 92,
          top: 1345,
          height: 46,
          borderRadius: '50%',
          backgroundColor: 'rgba(27,0,4,0.58)',
          filter: 'blur(22px)',
          opacity: interpolate(frame, [3, 12, 55, 63], [0, 0.8, 0.8, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
        }}
      />
    </AbsoluteFill>
  );
};
