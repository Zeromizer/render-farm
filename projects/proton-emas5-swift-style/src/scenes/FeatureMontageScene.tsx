import React from 'react';
import {Video} from '@remotion/media';
import {AbsoluteFill, Easing, Img, Interactive, Sequence, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {RacingBackground} from '../components/RacingBackground';

type FeatureCardProps = {
  durationInFrames: number;
  glyph: string;
  kicker: string;
  line1: string;
  line2?: string;
  car: string;
  accent?: string;
  videoTrimBefore?: number;
  videoPosition?: string;
  videoScale?: number;
  videoPlaybackRate?: number;
  videoMotionFrames?: number;
  videoFreezeImage?: string;
  stageImage?: string;
  stageImagePosition?: string;
  stageImageScale?: number;
};

const FeatureCard: React.FC<FeatureCardProps> = ({durationInFrames, glyph, kicker, line1, line2, car, accent = '#b60719', videoTrimBefore, videoPosition = '50% 50%', videoScale = 1.22, videoPlaybackRate = 1, videoMotionFrames, videoFreezeImage, stageImage, stageImagePosition = '50% 50%', stageImageScale = 1.04}) => {
  const frame = useCurrentFrame();
  const exitStart = durationInFrames - 12;
  const lastFrame = durationInFrames - 1;

  return (
    <AbsoluteFill>
      <RacingBackground />
      <Interactive.Div name="Ghost feature word" style={{position: 'absolute', top: 635, left: -40, right: -40, color: 'transparent', WebkitTextStroke: '2px rgba(255,255,255,0.14)', fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: 98, whiteSpace: 'nowrap', textAlign: 'center', rotate: '-4deg', translate: interpolate(frame, [0, lastFrame], ['-90px 0px', '100px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), opacity: interpolate(frame, [0, 10, exitStart, lastFrame], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>
        {line1} {line1}
      </Interactive.Div>
      <Interactive.Div name="Feature detail card" style={{position: 'absolute', top: 335, left: 170, width: 740, height: 390, borderRadius: 54, background: 'linear-gradient(135deg, #f6f7fb, #d8dde8)', border: '10px solid white', boxShadow: '0 28px 50px rgba(0,0,0,0.28)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: accent, fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: glyph.length > 5 ? 116 : 160, letterSpacing: 3, scale: interpolate(frame, [0, 18, exitStart, lastFrame], [0.72, 1, 1, 0.93], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: [Easing.spring({damping: 190}), Easing.linear, Easing.bezier(0.4, 0, 1, 1)], output: 'perceptual-scale'}), rotate: interpolate(frame, [0, 18, exitStart, lastFrame], ['-7deg', '0deg', '0deg', '2deg'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), opacity: interpolate(frame, [0, 7, exitStart + 2, lastFrame], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>
        {glyph}
        <div style={{position: 'absolute', bottom: -29, left: 48, right: 48, minHeight: 48, backgroundColor: accent, color: 'white', border: '5px solid white', borderRadius: 4, fontFamily: 'Arial Black, sans-serif', fontStyle: 'normal', fontSize: kicker.length > 28 ? 17 : 23, letterSpacing: kicker.length > 28 ? 0.4 : 1, whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '0 14px'}}>{kicker}</div>
      </Interactive.Div>
      <div style={{position: 'absolute', left: -80, right: -80, bottom: -70, height: 830, overflow: 'visible'}}>
        <div style={{position: 'absolute', inset: 0, clipPath: 'polygon(0 6%, 100% 0, 100% 100%, 0 100%)', background: 'linear-gradient(180deg, #3b3d43 0%, #17191e 57%, #090a0d 100%)', boxShadow: '0 -22px 60px rgba(0,44,66,0.34)'}} />
        {stageImage ? (
          <div style={{position: 'absolute', inset: 0, overflow: 'hidden', clipPath: 'polygon(0 6%, 100% 0, 100% 100%, 0 100%)'}}>
            <Img name="Authentic feature still" src={staticFile(stageImage)} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', objectPosition: stageImagePosition, scale: stageImageScale, filter: 'saturate(0.9) contrast(1.08) brightness(0.78)', opacity: interpolate(frame, [0, 8, exitStart + 2, lastFrame], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}} />
            <div style={{position: 'absolute', inset: 0, background: `linear-gradient(180deg, ${accent}44 0%, transparent 32%, transparent 57%, rgba(5,7,11,0.8) 100%)`, mixBlendMode: 'multiply'}} />
            <div style={{position: 'absolute', inset: 0, boxShadow: 'inset 0 0 90px rgba(0,0,0,0.48)'}} />
          </div>
        ) : videoTrimBefore === undefined ? (
          <>
            <div style={{position: 'absolute', left: 75, right: 75, top: 40, height: 520, borderRadius: '50%', background: `radial-gradient(ellipse, ${accent}66 0%, rgba(255,255,255,0.12) 35%, transparent 70%)`, filter: 'blur(20px)', opacity: interpolate(frame, [0, 18, exitStart - 2, lastFrame], [0, 0.9, 0.9, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}} />
            <div style={{position: 'absolute', left: 105, right: 105, bottom: 182, height: 86, borderRadius: '50%', background: 'radial-gradient(ellipse, rgba(0,0,0,0.88), rgba(0,0,0,0.2) 62%, transparent 76%)', filter: 'blur(16px)', opacity: interpolate(frame, [2, 18, exitStart + 1, lastFrame], [0, 0.94, 0.94, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}} />
            <Img name="Subtle vehicle floor reflection" src={staticFile(car)} style={{position: 'absolute', width: 930, height: 'auto', left: '50%', bottom: -115, translate: '-50% 0px', scale: '1 -0.28', transformOrigin: 'center bottom', opacity: interpolate(frame, [4, 18, exitStart - 2, lastFrame], [0, 0.12, 0.12, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), filter: 'blur(8px) saturate(0.65) brightness(0.55)', WebkitMaskImage: 'linear-gradient(to bottom, rgba(0,0,0,0.7), transparent 72%)'}} />
            <Img name="Staged Proton e.MAS 5 portrait" src={staticFile(car)} style={{position: 'absolute', width: 930, height: 'auto', left: '50%', bottom: 205, translate: interpolate(frame, [0, 20, exitStart, lastFrame], ['-50% 70px', '-50% 0px', '-50% 0px', '-50% 32px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)}), scale: interpolate(frame, [0, 24, exitStart, lastFrame], [0.91, 1, 1.015, 1.02], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1), output: 'perceptual-scale'}), opacity: interpolate(frame, [0, 9, exitStart + 2, lastFrame], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), filter: 'drop-shadow(0 30px 24px rgba(0,0,0,0.56)) brightness(0.98) saturate(0.92)'}} />
            <div style={{position: 'absolute', width: 400, height: 620, left: 90, top: -120, rotate: '-18deg', background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)', filter: 'blur(24px)', mixBlendMode: 'screen', translate: interpolate(frame, [5, exitStart - 2], ['-360px 0px', '900px 0px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), opacity: interpolate(frame, [5, 18, exitStart - 6, exitStart + 3], [0, 0.7, 0.7, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}} />
          </>
        ) : (
          <div style={{position: 'absolute', inset: 0, overflow: 'hidden', clipPath: 'polygon(0 6%, 100% 0, 100% 100%, 0 100%)'}}>
            <Video name="Authentic feature demonstration" src={staticFile('footage/emas5-freedom-1080p.mp4')} trimBefore={videoTrimBefore} durationInFrames={videoMotionFrames ?? durationInFrames} playbackRate={videoPlaybackRate} muted objectFit="cover" style={{position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', objectPosition: videoPosition, scale: videoScale, filter: 'saturate(0.88) contrast(1.08) brightness(0.78)', opacity: interpolate(frame, [0, 8, exitStart + 2, lastFrame], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}} />
            {videoFreezeImage && videoMotionFrames ? <Img name="Clean held feature detail" src={staticFile(videoFreezeImage)} from={videoMotionFrames - 1} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', objectPosition: videoPosition, scale: videoScale, filter: 'saturate(0.88) contrast(1.08) brightness(0.78)', opacity: interpolate(frame, [0, 8, exitStart + 2, lastFrame], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}} /> : null}
            <div style={{position: 'absolute', inset: 0, background: `linear-gradient(180deg, ${accent}55 0%, transparent 32%, transparent 57%, rgba(5,7,11,0.84) 100%)`, mixBlendMode: 'multiply'}} />
            <div style={{position: 'absolute', inset: 0, boxShadow: 'inset 0 0 90px rgba(0,0,0,0.52)'}} />
          </div>
        )}
      </div>
      <Interactive.Div name="Feature headline" style={{position: 'absolute', bottom: 98, left: 80, right: 80, color: 'white', textAlign: 'center', fontFamily: 'Impact, Arial Black, sans-serif', fontStyle: 'italic', fontSize: 78, lineHeight: 0.94, letterSpacing: 2, textShadow: '0 7px 0 rgba(0,0,0,0.24)', opacity: interpolate(frame, [9, 22, exitStart + 1, lastFrame], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: [Easing.bezier(0.16, 1, 0.3, 1), Easing.linear, Easing.bezier(0.4, 0, 1, 1)]}), translate: interpolate(frame, [7, 24, exitStart + 1, lastFrame], ['0px 78px', '0px 0px', '0px 0px', '0px -35px'], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1)})}}>
        {line1}{line2 ? <><br />{line2}</> : null}
      </Interactive.Div>
      <div style={{position: 'absolute', bottom: 42, left: 465, display: 'flex', gap: 14, opacity: interpolate(frame, [18, 28, exitStart + 1, lastFrame], [0, 0.48, 0.48, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}), color: 'white', fontSize: 24}}><span>&lsaquo;</span><span>&lsaquo;</span><span>&lsaquo;</span></div>
    </AbsoluteFill>
  );
};

export const FeatureMontageScene: React.FC = () => (
  <AbsoluteFill>
    <Sequence name="Up to 325 km WLTP range" durationInFrames={68} premountFor={18}><FeatureCard durationInFrames={68} glyph="325 KM" kicker="UP TO 325 KM WLTP RANGE" line1="GO FURTHER" line2="EVERY DAY" car="emas5/front-right-v8.png" videoTrimBefore={5580} videoPosition="52% 50%" /></Sequence>
    <Sequence name="21-minute DC fast charge" from={68} durationInFrames={67} premountFor={18}><FeatureCard durationInFrames={67} glyph="21 MIN" kicker="DC CHARGING: 30%-80%" line1="FAST DC" line2="CHARGING" car="emas5/front-left-v8.png" accent="#0069a6" /></Sequence>
    <Sequence name="14.6-inch display" from={135} durationInFrames={68} premountFor={18}><FeatureCard durationInFrames={68} glyph={'14.6\"'} kicker="14.6-INCH FULL HD DISPLAY HEAD UNIT" line1="14.6-INCH" line2="FULL HD" car="emas5/front-v8.png" accent="#1b637c" videoTrimBefore={3282} videoPosition="50% 52%" /></Sequence>
    <Sequence name="375-litre boot" from={203} durationInFrames={67} premountFor={18}><FeatureCard durationInFrames={67} glyph="375 L" kicker="EVERYDAY BOOT SPACE" line1="375-LITRE" line2="BOOT SPACE" car="emas5/rear-three-quarter-v8.png" accent="#90510d" videoTrimBefore={6864} videoPosition="50% 50%" /></Sequence>
    <Sequence name="70-litre frunk" from={270} durationInFrames={68} premountFor={18}><FeatureCard durationInFrames={68} glyph="70 L" kicker="FRONT STORAGE COMPARTMENT" line1="70-LITRE" line2="FRUNK" car="emas5/front-right-v8.png" accent="#8c3c14" stageImage="footage/frunk-70l.jpg" stageImagePosition="50% 48%" /></Sequence>
    <Sequence name="Rear-wheel drive" from={338} durationInFrames={67} premountFor={18}><FeatureCard durationInFrames={67} glyph="RWD" kicker="REAR-WHEEL DRIVE" line1="CONFIDENT" line2="HANDLING" car="emas5/side.png" accent="#006f83" videoTrimBefore={5520} videoPosition="50% 50%" /></Sequence>
    <Sequence name="Six airbags and ADAS" from={405} durationInFrames={68} premountFor={18}><FeatureCard durationInFrames={68} glyph="6" kicker="6 AIRBAGS + ADAS" line1="TOP-TIER" line2="SAFETY" car="emas5/front-left-v8.png" accent="#174a74" /></Sequence>
  </AbsoluteFill>
);
