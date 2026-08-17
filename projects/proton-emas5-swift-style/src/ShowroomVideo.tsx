import React from 'react';
import {AbsoluteFill, Sequence, Series, staticFile} from 'remotion';
import {Audio} from '@remotion/media';
import {BrandBar} from './components/BrandBar';
import {HookScene} from './scenes/HookScene';
import {RevealScene} from './scenes/RevealScene';
import {FeatureMontageScene} from './scenes/FeatureMontageScene';
import {OfferScene} from './scenes/OfferScene';
import {OutroScene} from './scenes/OutroScene';

export const ProtonEmas5Showroom: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#b50716'}}>
      <Series>
        <Series.Sequence name="Electric and practical hook" durationInFrames={60}><HookScene /></Series.Sequence>
        <Series.Sequence name="e.MAS 5 reveal" durationInFrames={120}><RevealScene /></Series.Sequence>
        <Series.Sequence name="Seven selling points" durationInFrames={360}><FeatureMontageScene /></Series.Sequence>
        <Series.Sequence name="Range call to action" durationInFrames={60}><OfferScene /></Series.Sequence>
        <Series.Sequence name="Proton e.MAS outro" durationInFrames={60}><OutroScene /></Series.Sequence>
      </Series>
      <Sequence name="Persistent Proton e.MAS brand bar" durationInFrames={600}><BrandBar /></Sequence>
      <Audio src={staticFile('reference/43n6xu.mp4')} volume={0.92} />
    </AbsoluteFill>
  );
};
