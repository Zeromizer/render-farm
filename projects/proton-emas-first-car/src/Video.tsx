import React from 'react';
import {AbsoluteFill, Sequence, staticFile} from 'remotion';
import {Audio, Video} from '@remotion/media';

const INK = '#05090B';

const Footage: React.FC<{src: string; trimBefore: number; name: string; position?: string; dark?: number}> = ({src, trimBefore, name, position = '50% 50%', dark = .12}) => <AbsoluteFill style={{backgroundColor: INK}}><Video name={name} src={staticFile(src)} trimBefore={trimBefore} muted style={{width: '100%', height: '100%', objectFit: 'cover', objectPosition: position}} /><AbsoluteFill style={{background: `linear-gradient(180deg,rgba(2,5,6,.2),transparent 42%,rgba(2,5,6,${dark}) 100%)`}} /></AbsoluteFill>;

const PortraitAsLandscape: React.FC<{src: string; trimBefore: number; name: string; y?: number}> = ({src, trimBefore, name, y = -50}) => <AbsoluteFill style={{backgroundColor:INK,overflow:'hidden'}}><Video name={name} src={staticFile(src)} trimBefore={trimBefore} muted style={{position:'absolute',width:'100%',height:'auto',left:0,top:'50%',translate:`0 ${y}%`}}/><AbsoluteFill style={{background:'linear-gradient(180deg,rgba(2,5,6,.14),transparent 46%,rgba(2,5,6,.18) 100%)'}}/></AbsoluteFill>;

const Panel: React.FC<{children:React.ReactNode}> = ({children}) => <div style={{width:'100%',height:'33.333%',flexShrink:0,overflow:'hidden',position:'relative',borderBottom:`5px solid ${INK}`}}>{children}<div style={{position:'absolute',left:0,right:0,bottom:0,height:22,background:'linear-gradient(transparent,rgba(0,0,0,.68))'}}/></div>;
const TriplePanel: React.FC<{top: React.ReactNode; middle: React.ReactNode; bottom: React.ReactNode}> = ({top,middle,bottom}) => <AbsoluteFill style={{backgroundColor:INK}}><Panel>{top}</Panel><Panel>{middle}</Panel><Panel>{bottom}</Panel></AbsoluteFill>;

const Intro: React.FC = () => <Footage src="emas7/front-reveal.mp4" trimBefore={8} name="Opening e.MAS 7 detail" position="48% 50%" dark={.62}/>;
const FivePanels: React.FC = () => <TriplePanel top={<Footage src="emas5-video/freedom.mp4" trimBefore={2360} name="e.MAS 5 infotainment" position="50% 50%" />} middle={<Footage src="emas5-video/freedom.mp4" trimBefore={2840} name="e.MAS 5 rolling hero" position="50% 50%" />} bottom={<Footage src="emas5-video/freedom.mp4" trimBefore={1015} name="e.MAS 5 rear detail" position="50% 50%" />} />;
const SevenPanels: React.FC = () => <TriplePanel top={<PortraitAsLandscape src="emas7-requested/top-115-optimized.mp4" trimBefore={24} name="e.MAS 7 cockpit - top clip 115" />} middle={<PortraitAsLandscape src="emas7-requested/center-25-optimized.mp4" trimBefore={24} name="e.MAS 7 front - centre clip 25" />} bottom={<PortraitAsLandscape src="emas7-requested/bottom-24-optimized.mp4" trimBefore={24} name="e.MAS 7 cargo - bottom clip 24" />} />;
const Lifestyle: React.FC = () => <Footage src="emas5-video/freedom.mp4" trimBefore={3135} name="e.MAS 5 city drive" position="50% 50%" dark={.55}/>;
const SevenFeature: React.FC = () => <AbsoluteFill style={{backgroundColor:INK}}><div style={{position:'absolute',left:0,right:0,top:610,height:700,overflow:'hidden'}}><PortraitAsLandscape src="emas7/charging.mp4" trimBefore={5} name="e.MAS 7 charging landscape feature"/></div></AbsoluteFill>;
const Choice: React.FC = () => <AbsoluteFill style={{backgroundColor:INK}}><div style={{height:'50%',position:'relative',overflow:'hidden'}}><Footage src="emas5-video/freedom.mp4" trimBefore={1050} name="e.MAS 5 hero choice" position="50% 50%" dark={.28}/></div><div style={{height:'50%',position:'relative',overflow:'hidden',borderTop:`5px solid ${INK}`}}><Footage src="emas7-landscape/power-up.mp4" trimBefore={60} name="e.MAS 7 moving Singapore driving montage" position="50% 50%" dark={.3}/></div></AbsoluteFill>;
const Cta: React.FC = () => <AbsoluteFill style={{backgroundColor:'black'}}><Video src={staticFile('cta/proton-logo.mp4')} muted style={{width:'100%',height:'100%',objectFit:'cover'}}/></AbsoluteFill>;

export const FirstCarVideo: React.FC = () => <AbsoluteFill style={{backgroundColor:INK}}>
  <Sequence durationInFrames={125}><Intro/></Sequence>
  <Sequence from={125} durationInFrames={145}><FivePanels/></Sequence>
  <Sequence from={270} durationInFrames={120}><SevenPanels/></Sequence>
  <Sequence from={390} durationInFrames={120}><Lifestyle/></Sequence>
  <Sequence from={510} durationInFrames={100}><SevenFeature/></Sequence>
  <Sequence from={610} durationInFrames={150}><Choice/></Sequence>
  <Sequence from={760} durationInFrames={140}><Cta/></Sequence>
  <Audio src={staticFile('audio/music.wav')} volume={.22}/>
</AbsoluteFill>;
