import React from 'react';
import {AbsoluteFill, Sequence, Series, interpolate, staticFile} from 'remotion';
import {Audio} from '@remotion/media';
import {BrandBar} from './components/BrandBar';
import {HookScene} from './scenes/HookScene';
import {RevealScene} from './scenes/RevealScene';
import {FeatureMontageScene} from './scenes/FeatureMontageScene';
import {OfferScene} from './scenes/OfferScene';
import {OutroScene} from './scenes/OutroScene';
import {DealerCtaScene} from './scenes/DealerCtaScene';

export const ProtonEmas5Showroom: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#b50716'}}>
      <Series>
        <Series.Sequence name="Electric and practical hook" durationInFrames={60}><HookScene /></Series.Sequence>
        <Series.Sequence name="e.MAS 5 reveal" durationInFrames={158}><RevealScene /></Series.Sequence>
        <Series.Sequence name="Seven selling points" durationInFrames={473}><FeatureMontageScene /></Series.Sequence>
        <Series.Sequence name="Range call to action" durationInFrames={67}><OfferScene /></Series.Sequence>
        <Series.Sequence name="Proton e.MAS outro" durationInFrames={45}><OutroScene /></Series.Sequence>
        <Series.Sequence name="Dealer call to action" durationInFrames={150}><DealerCtaScene /></Series.Sequence>
      </Series>
      <Sequence name="Persistent Proton e.MAS brand bar" durationInFrames={758} premountFor={30} style={{zIndex: 100}}><BrandBar /></Sequence>
      <Audio src={staticFile('audio/electric-shores.mp3')} volume={(frame) => interpolate(frame, [0, 15, 908, 952], [0, 0.82, 0.82, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})} />
    </AbsoluteFill>
  );
};
