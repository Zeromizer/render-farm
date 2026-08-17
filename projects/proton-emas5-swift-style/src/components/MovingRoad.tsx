import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';

export const MovingRoad: React.FC<{opacity?: number}> = ({opacity = 1}) => {
  const frame = useCurrentFrame();
  const dashOffset = -frame * 22;
  const arrowTravel = interpolate(frame % 30, [0, 29], [0, 150]);

  return (
    <AbsoluteFill style={{opacity, overflow: 'hidden'}}>
      <div style={{position: 'absolute', left: 0, right: 0, bottom: 0, height: 270, background: 'linear-gradient(135deg, #25272c, #17191d)'}} />
      <svg
        viewBox="0 0 1080 920"
        style={{position: 'absolute', left: 0, right: 0, bottom: -70, width: 1080, height: 920}}
      >
        <defs>
          <linearGradient id="roadShade" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#393b40" />
            <stop offset="0.58" stopColor="#25272c" />
            <stop offset="1" stopColor="#17191d" />
          </linearGradient>
          <filter id="roadGlow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="10" />
          </filter>
        </defs>
        <path
          d="M 115 1030 C 15 710 105 455 390 335 C 650 225 920 265 1240 500"
          fill="none"
          stroke="rgba(255,255,255,0.9)"
          strokeWidth="742"
          strokeLinecap="round"
        />
        <path
          d="M 115 1030 C 15 710 105 455 390 335 C 650 225 920 265 1240 500"
          fill="none"
          stroke="url(#roadShade)"
          strokeWidth="716"
          strokeLinecap="round"
        />
        <path
          d="M 115 1030 C 15 710 105 455 390 335 C 650 225 920 265 1240 500"
          fill="none"
          stroke="rgba(255,255,255,0.16)"
          strokeWidth="10"
          strokeDasharray="92 82"
          strokeDashoffset={dashOffset}
        />
        <path
          d="M 80 1010 C 20 750 125 535 365 430 C 575 338 800 350 1075 515"
          fill="none"
          stroke="rgba(255,255,255,0.09)"
          strokeWidth="26"
          strokeDasharray="22 118"
          strokeDashoffset={dashOffset * 1.6}
          filter="url(#roadGlow)"
        />
        <g
          opacity="0.42"
          stroke="white"
          strokeWidth="12"
          fill="none"
          transform={`translate(0 ${arrowTravel})`}
        >
          <path d="M 400 640 l 48 34 -48 34" />
          <path d="M 465 640 l 48 34 -48 34" />
          <path d="M 530 640 l 48 34 -48 34" />
        </g>
      </svg>
      <div
        style={{
          position: 'absolute',
          left: 75,
          right: 95,
          bottom: 70,
          height: 180,
          background: 'linear-gradient(180deg, transparent, rgba(0,0,0,0.32))',
          filter: 'blur(26px)',
        }}
      />
    </AbsoluteFill>
  );
};
