import React from 'react';
import {AbsoluteFill, Easing, Img, Interactive, Sequence, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {RacingBackground} from '../components/RacingBackground';

const CARD_DURATION = 66;

type FeatureCardProps = {
  glyph: string;
  kicker: string;
  line1: string;
  line2?: string;
  car: string;
  accent?: string;
};

const FeatureCard: React.FC<FeatureCardProps> = ({glyph, kicker, line1, line2, car, accent = '#b60719'}) => {
  const frame = useCurrentFrame();
  const suspension = Math.sin(frame * 0.24) * 3.5;

  return (
    <AbsoluteFill>
      <RacingBackground darkRoad />
      <Interactive.Div name="Ghost feature word" style={{position: 'absolute', top: 535, left: -40, right: -40, color: 'transparent', WebkitTextStroke: '2px rgba(255,255,255,0.14)', fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: 98, whiteSpace: 'nowrap', textAlign: 'center', rotate: '-4deg', translate: interpolate(frame, [0, CARD_DURATION - 1], ['-90px 0px', '100px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), opacity: interpolate(frame, [0, 10, 54, 65], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>
        {line1} {line1}
      </Interactive.Div>
      <Interactive.Div name="Feature detail card" style={{position: 'absolute', top: 235, left: 170, width: 740, height: 390, borderRadius: 54, background: 'linear-gradient(135deg, #f6f7fb, #d8dde8)', border: '10px solid white', boxShadow: '0 28px 50px rgba(0,0,0,0.28)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: accent, fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: glyph.length > 5 ? 116 : 160, letterSpacing: 3, scale: interpolate(frame, [0, 18, 54, 65], [0.72, 1, 1, 0.93], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: [Easing.spring({damping: 190}), Easing.linear, Easing.bezier(0.4, 0, 1, 1)], output: 'perceptual-scale'}), rotate: interpolate(frame, [0, 18, 54, 65], ['-7deg', '0deg', '0deg', '2deg'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), opacity: interpolate(frame, [0, 7, 56, 65], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>
        {glyph}
        <div style={{position: 'absolute', bottom: -29, left: 100, right: 100, minHeight: 48, backgroundColor: accent, color: 'white', border: '5px solid white', borderRadius: 4, fontFamily: 'Arial Black, sans-serif', fontStyle: 'normal', fontSize: 23, letterSpacing: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '0 14px'}}>{kicker}</div>
      </Interactive.Div>
      <div style={{position: 'absolute', inset: 0, translate: `0px ${suspension}px`}}>
        <div style={{position: 'absolute', left: 105, bottom: 255, width: 860, height: 72, borderRadius: '50%', backgroundColor: 'rgba(5,7,10,0.62)', filter: 'blur(24px)', opacity: interpolate(frame, [2, 18, 55, 65], [0, 0.9, 0.9, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), scale: interpolate(frame, [0, 22, 65], [0.6, 1, 1.08], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', output: 'perceptual-scale'})}} />
        <Img name="Directional vehicle trail" src={staticFile(car)} style={{position: 'absolute', width: 930, height: 'auto', left: '50%', bottom: 225, translate: interpolate(frame, [0, 20, 54, 65], ['-980px 0px', '-505px 0px', '-505px 0px', '40px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)}), opacity: interpolate(frame, [0, 10, 51, 65], [0.26, 0.03, 0.03, 0.28], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), filter: 'blur(11px) brightness(1.08)', scale: '1.045 1'}} />
        <Img name="Feature car angle" src={staticFile(car)} style={{position: 'absolute', width: 930, height: 'auto', left: '50%', bottom: 225, translate: interpolate(frame, [0, 20, 54, 65], ['-950px 0px', '-465px 0px', '-465px 0px', '120px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)}), scale: interpolate(frame, [0, 48, 65], [0.88, 1.035, 1.055], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1), output: 'perceptual-scale'}), rotate: interpolate(frame, [0, 22, 54, 65], ['-1.8deg', '0deg', '0deg', '1.2deg'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), filter: 'drop-shadow(0 30px 22px rgba(0,0,0,0.48)) brightness(1.025)'}} />
      </div>
      <Interactive.Div name="Feature headline" style={{position: 'absolute', bottom: 98, left: 80, right: 80, color: 'white', textAlign: 'center', fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: 78, lineHeight: 0.94, letterSpacing: 2, textShadow: '0 7px 0 rgba(0,0,0,0.24)', opacity: interpolate(frame, [9, 22, 55, 65], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: [Easing.bezier(0.16, 1, 0.3, 1), Easing.linear, Easing.bezier(0.4, 0, 1, 1)]}), translate: interpolate(frame, [7, 24, 55, 65], ['0px 78px', '0px 0px', '0px 0px', '0px -35px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)})}}>
        {line1}{line2 ? <><br />{line2}</> : null}
      </Interactive.Div>
      <div style={{position: 'absolute', bottom: 42, left: 465, display: 'flex', gap: 14, opacity: interpolate(frame, [18, 28, 55, 65], [0, 0.48, 0.48, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), color: 'white', fontSize: 24}}><span>â€¹</span><span>â€¹</span><span>â€¹</span></div>
    </AbsoluteFill>
  );
};

export const FeatureMontageScene: React.FC = () => (
  <AbsoluteFill>
    <Sequence name="Up to 325 km WLTP range" durationInFrames={CARD_DURATION} premountFor={18}><FeatureCard glyph="325 KM" kicker="UP TO 325 KM WLTP RANGE" line1="GO FARTHER" line2="EVERY DAY" car="emas5/front-right.png" /></Sequence>
    <Sequence name="21-minute DC fast charge" from={66} durationInFrames={CARD_DURATION} premountFor={18}><FeatureCard glyph="21 MIN" kicker="DC CHARGING: 30%â€“80%" line1="FAST DC" line2="CHARGING" car="emas5/front-left.png" accent="#0069a6" /></Sequence>
    <Sequence name="14.6-inch display" from={132} durationInFrames={CARD_DURATION} premountFor={18}><FeatureCard glyph="14.6â€" kicker="FHD DISPLAY HEAD UNIT" line1="14.6-INCH" line2="FHD DISPLAY" car="emas5/front.png" accent="#1b637c" /></Sequence>
    <Sequence name="375-litre boot" from={198} durationInFrames={CARD_DURATION} premountFor={18}><FeatureCard glyph="375 L" kicker="EVERYDAY BOOT SPACE" line1="375-LITRE" line2="BOOT SPACE" car="emas5/rear-three-quarter.png" accent="#90510d" /></Sequence>
    <Sequence name="70-litre frunk" from={264} durationInFrames={CARD_DURATION} premountFor={18}><FeatureCard glyph="70 L" kicker="FRONT STORAGE COMPARTMENT" line1="70-LITRE" line2="FRUNK" car="emas5/front-right.png" accent="#8c3c14" /></Sequence>
    <Sequence name="Rear-wheel drive" from={330} durationInFrames={CARD_DURATION} premountFor={18}><FeatureCard glyph="RWD" kicker="REAR-WHEEL DRIVE" line1="CONFIDENT" line2="HANDLING" car="emas5/side.png" accent="#006f83" /></Sequence>
    <Sequence name="Six airbags and ADAS" from={396} durationInFrames={CARD_DURATION} premountFor={18}><FeatureCard glyph="6" kicker="6 AIRBAGS + ADAS" line1="TOP-TIER" line2="SAFETY" car="emas5/front-left.png" accent="#174a74" /></Sequence>
  </AbsoluteFill>
);
