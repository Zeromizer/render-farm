import React from 'react';
import {AbsoluteFill, Easing, Img, Interactive, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {RacingBackground} from '../components/RacingBackground';

const FeatureCard: React.FC<{glyph: string; kicker: string; line1: string; line2?: string; car: string; accent?: string}> = ({glyph, kicker, line1, line2, car, accent = '#b60719'}) => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  return (
    <AbsoluteFill>
      <RacingBackground darkRoad />
      <Interactive.Div name="Ghost feature word" style={{position: 'absolute', top: 535, left: -40, right: -40, color: 'transparent', WebkitTextStroke: '2px rgba(255,255,255,0.14)', fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: 98, whiteSpace: 'nowrap', textAlign: 'center', rotate: '-4deg', translate: interpolate(frame, [0, durationInFrames - 1], ['-80px 0px', '80px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>{line1} {line1}</Interactive.Div>
      <Interactive.Div name="Feature detail card" style={{position: 'absolute', top: 235, left: 170, width: 740, height: 390, borderRadius: 54, background: 'linear-gradient(135deg, #f6f7fb, #d8dde8)', border: '10px solid white', boxShadow: '0 28px 50px rgba(0,0,0,0.28)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: accent, fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: glyph.length > 5 ? 116 : 160, letterSpacing: 3, scale: interpolate(frame, [0, 12], [0.72, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.spring({damping: 180}), output: 'perceptual-scale'}), rotate: interpolate(frame, [0, 12], ['-7deg', '0deg'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>
        {glyph}
        <div style={{position: 'absolute', bottom: -29, left: 100, right: 100, minHeight: 48, backgroundColor: accent, color: 'white', border: '5px solid white', borderRadius: 4, fontFamily: 'Arial Black, sans-serif', fontStyle: 'normal', fontSize: 23, letterSpacing: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '0 14px'}}>{kicker}</div>
      </Interactive.Div>
      <Img name="Feature car angle" src={staticFile(car)} style={{position: 'absolute', width: 930, height: 'auto', left: '50%', bottom: 225, translate: interpolate(frame, [0, 15, durationInFrames - 10, durationInFrames], ['-920px 0px', '-465px 0px', '-465px 0px', '120px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)}), scale: interpolate(frame, [0, durationInFrames - 1], [0.9, 1.02], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', output: 'perceptual-scale'}), filter: 'drop-shadow(0 30px 22px rgba(0,0,0,0.5))'}} />
      <Interactive.Div name="Feature headline" style={{position: 'absolute', bottom: 98, left: 80, right: 80, color: 'white', textAlign: 'center', fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: 78, lineHeight: 0.94, letterSpacing: 2, textShadow: '0 7px 0 rgba(0,0,0,0.24)', opacity: interpolate(frame, [7, 15], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), translate: interpolate(frame, [5, 16], ['0px 70px', '0px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>{line1}{line2 ? <><br />{line2}</> : null}</Interactive.Div>
      <div style={{position: 'absolute', bottom: 42, left: 465, display: 'flex', gap: 14, opacity: 0.48, color: 'white', fontSize: 24}}><span>《</span><span>《</span><span>《</span></div>
    </AbsoluteFill>
  );
};

export const FeatureMontageScene: React.FC = () => (
  <AbsoluteFill>
    <Sequence name="Up to 325 km WLTP range" durationInFrames={51}><FeatureCard glyph="325 KM" kicker="UP TO 325 KM WLTP RANGE" line1="GO FARTHER" line2="EVERY DAY" car="emas5/front-right.png" /></Sequence>
    <Sequence name="21-minute DC fast charge" from={51} durationInFrames={51}><FeatureCard glyph="21 MIN" kicker="DC CHARGING: 30%–80%" line1="FAST DC" line2="CHARGING" car="emas5/front-left.png" accent="#0069a6" /></Sequence>
    <Sequence name="14.6-inch display" from={102} durationInFrames={51}><FeatureCard glyph="14.6”" kicker="FHD DISPLAY HEAD UNIT" line1="14.6-INCH" line2="FHD DISPLAY" car="emas5/front.png" accent="#1b637c" /></Sequence>
    <Sequence name="375-litre boot" from={153} durationInFrames={51}><FeatureCard glyph="375 L" kicker="EVERYDAY BOOT SPACE" line1="375-LITRE" line2="BOOT SPACE" car="emas5/rear-three-quarter.png" accent="#90510d" /></Sequence>
    <Sequence name="70-litre frunk" from={204} durationInFrames={51}><FeatureCard glyph="70 L" kicker="FRONT STORAGE COMPARTMENT" line1="70-LITRE" line2="FRUNK" car="emas5/front-right.png" accent="#8c3c14" /></Sequence>
    <Sequence name="Rear-wheel drive" from={255} durationInFrames={51}><FeatureCard glyph="RWD" kicker="REAR-WHEEL DRIVE" line1="CONFIDENT" line2="HANDLING" car="emas5/side.png" accent="#006f83" /></Sequence>
    <Sequence name="Six airbags and ADAS" from={306} durationInFrames={54}><FeatureCard glyph="6" kicker="6 AIRBAGS + ADAS" line1="TOP-TIER" line2="SAFETY" car="emas5/front-left.png" accent="#174a74" /></Sequence>
  </AbsoluteFill>
);
