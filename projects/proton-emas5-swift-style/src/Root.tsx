import React from 'react';
import {Composition, Folder} from 'remotion';
import {ProtonEmas5Showroom} from './ShowroomVideo';
import {HookScene} from './scenes/HookScene';
import {RevealScene} from './scenes/RevealScene';
import {FeatureMontageScene} from './scenes/FeatureMontageScene';
import {OfferScene} from './scenes/OfferScene';
import {OutroScene} from './scenes/OutroScene';
import {DealerCtaScene} from './scenes/DealerCtaScene';

export const Root: React.FC = () => {
  return (
    <>
      <Folder name="Proton-eMAS-5-scenes">
        <Composition id="HookScene" component={HookScene} durationInFrames={60} fps={30} width={1080} height={1920} />
        <Composition id="RevealScene" component={RevealScene} durationInFrames={158} fps={30} width={1080} height={1920} />
        <Composition id="FeatureMontage" component={FeatureMontageScene} durationInFrames={473} fps={30} width={1080} height={1920} />
        <Composition id="OfferScene" component={OfferScene} durationInFrames={67} fps={30} width={1080} height={1920} />
        <Composition id="OutroScene" component={OutroScene} durationInFrames={45} fps={30} width={1080} height={1920} />
        <Composition id="DealerCtaScene" component={DealerCtaScene} durationInFrames={150} fps={30} width={1080} height={1920} />
      </Folder>
      <Composition id="ProtonEmas5Showroom" component={ProtonEmas5Showroom} durationInFrames={953} fps={30} width={1080} height={1920} />
    </>
  );
};
