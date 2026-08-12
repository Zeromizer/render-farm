import React from 'react';
import {Composition, Folder} from 'remotion';
import {FirstCarVideo} from './Video';
import {HookScene} from './scenes/HookScene';
import {ChoiceScene} from './scenes/ChoiceScene';
import {BenefitsScene} from './scenes/BenefitsScene';
import {FirstCarScene} from './scenes/FirstCarScene';
import {CtaScene} from './scenes/CtaScene';

export const Root: React.FC = () => (
  <>
    <Folder name="First-car-scenes">
      <Composition id="Hook" component={HookScene} durationInFrames={120} fps={30} width={1080} height={1920} />
      <Composition id="TwoChoices" component={ChoiceScene} durationInFrames={180} fps={30} width={1080} height={1920} />
      <Composition id="Benefits" component={BenefitsScene} durationInFrames={330} fps={30} width={1080} height={1920} />
      <Composition id="FirstCarMessage" component={FirstCarScene} durationInFrames={150} fps={30} width={1080} height={1920} />
      <Composition id="CTA" component={CtaScene} durationInFrames={120} fps={30} width={1080} height={1920} />
    </Folder>
    <Composition id="ProtonEmasFirstCar" component={FirstCarVideo} durationInFrames={900} fps={30} width={1080} height={1920} />
  </>
);
