import React from 'react';
import {AbsoluteFill, Sequence, staticFile} from 'remotion';
import {Audio} from '@remotion/media';
import {CaptionLayer} from './components';
import {HookScene} from './scenes/HookScene';
import {ChoiceScene} from './scenes/ChoiceScene';
import {BenefitsScene} from './scenes/BenefitsScene';
import {FirstCarScene} from './scenes/FirstCarScene';
import {CtaScene} from './scenes/CtaScene';

export const FirstCarVideo: React.FC = () => <AbsoluteFill style={{backgroundColor: '#071114'}}><Sequence name="Hook" durationInFrames={120}><HookScene /></Sequence><Sequence name="Two cars, two needs" from={120} durationInFrames={180}><ChoiceScene /></Sequence><Sequence name="Why e.MAS works" from={300} durationInFrames={330}><BenefitsScene /></Sequence><Sequence name="First-car choice" from={630} durationInFrames={150}><FirstCarScene /></Sequence><Sequence name="Call to action" from={780} durationInFrames={120}><CtaScene /></Sequence><Audio name="Original electronic score" src={staticFile('audio/music.wav')} volume={.2} /><Sequence from={0} durationInFrames={120}><Audio src={staticFile('audio/voiceover/hook.mp3')} volume={1} playbackRate={1.25} /></Sequence><Sequence from={120} durationInFrames={180}><Audio src={staticFile('audio/voiceover/choices.mp3')} volume={1} playbackRate={1.35} /></Sequence><Sequence from={300} durationInFrames={330}><Audio src={staticFile('audio/voiceover/benefits.mp3')} volume={1} /></Sequence><Sequence from={630} durationInFrames={150}><Audio src={staticFile('audio/voiceover/first-car.mp3')} volume={1} playbackRate={1.05} /></Sequence><Sequence from={780} durationInFrames={120}><Audio src={staticFile('audio/voiceover/cta.mp3')} volume={1} /></Sequence><Sequence from={113} durationInFrames={36}><Audio src={staticFile('audio/whoosh.wav')} volume={.32} /></Sequence><Sequence from={293} durationInFrames={24}><Audio src={staticFile('audio/impact.wav')} volume={.28} /></Sequence><Sequence from={773} durationInFrames={24}><Audio src={staticFile('audio/impact.wav')} volume={.24} /></Sequence><CaptionLayer /></AbsoluteFill>;
