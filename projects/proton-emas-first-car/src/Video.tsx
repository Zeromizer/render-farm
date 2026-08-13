import React from 'react';
import {AbsoluteFill, Easing, Interactive, interpolate, Sequence, staticFile, useCurrentFrame} from 'remotion';
import {Audio, Video} from '@remotion/media';
import {BODY_FONT, DISPLAY_FONT, FontStyles} from './fonts';

const ACCENT = '#78E7D0';
const WHITE = '#F7F5F0';
const INK = '#05090B';

const Footage: React.FC<{src: string; trimBefore: number; name: string; position?: string; dark?: number}> = ({src, trimBefore, name, position = '50% 50%', dark = .12}) => <AbsoluteFill style={{backgroundColor: INK}}><Video name={name} src={staticFile(src)} trimBefore={trimBefore} muted style={{width: '100%', height: '100%', objectFit: 'cover', objectPosition: position}} /><AbsoluteFill style={{background: `linear-gradient(180deg,rgba(2,5,6,.2),transparent 42%,rgba(2,5,6,${dark}) 100%)`}} /></AbsoluteFill>;

const PortraitAsLandscape: React.FC<{src: string; trimBefore: number; name: string; y?: number}> = ({src, trimBefore, name, y = -50}) => <AbsoluteFill style={{backgroundColor:INK,overflow:'hidden'}}><Video name={name} src={staticFile(src)} trimBefore={trimBefore} muted style={{position:'absolute',width:'100%',height:'auto',left:0,top:'50%',translate:`0 ${y}%`}}/><AbsoluteFill style={{background:'linear-gradient(180deg,rgba(2,5,6,.14),transparent 46%,rgba(2,5,6,.18) 100%)'}}/></AbsoluteFill>;

type KineticBeat = {text: string; accent?: boolean; outline?: boolean};

const KineticWords: React.FC<{beats: KineticBeat[]; top?: number; size?: number; align?: 'left'|'center'; delay?: number}> = ({beats, top = 125, size = 112, align = 'center', delay = 0}) => {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{alignItems: align === 'center' ? 'center' : 'flex-start', padding:`${top}px 72px 0`, textAlign:align, color:WHITE, pointerEvents:'none'}}>
    {beats.map((beat, index) => {
      const start = delay + index * 11;
      return <Interactive.Div key={`${beat.text}-${index}`} name={`Kinetic word: ${beat.text}`} style={{
        display:'inline-block', width:'fit-content', fontFamily:DISPLAY_FONT, fontSize:size, lineHeight:.82, fontWeight:700,
        letterSpacing:.5, textTransform:'uppercase', color:beat.outline ? 'transparent' : beat.accent ? INK : WHITE,
        WebkitTextStroke:beat.outline ? `3px ${ACCENT}` : undefined, backgroundColor:beat.accent ? ACCENT : 'transparent',
        padding:beat.accent ? '8px 18px 13px' : '0 4px', marginTop:index === 0 ? 0 : 16,
        boxShadow:beat.accent ? '0 14px 38px rgba(0,0,0,.28)' : undefined,
        textShadow:beat.accent || beat.outline ? undefined : '0 5px 34px rgba(0,0,0,.75)',
        opacity:interpolate(frame,[start,start+7],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp'}),
        translate:interpolate(frame,[start,start+12],[index % 2 === 0 ? '-90px 0px' : '90px 0px','0px 0px'],{easing:Easing.bezier(.16,1,.3,1),extrapolateLeft:'clamp',extrapolateRight:'clamp'}),
        scale:interpolate(frame,[start,start+12],[1.22,1],{easing:Easing.bezier(.16,1,.3,1),extrapolateLeft:'clamp',extrapolateRight:'clamp'})
      }}>{beat.text}</Interactive.Div>;
    })}
  </AbsoluteFill>;
};

const ModelStamp: React.FC<{model: string; descriptor: string; top?: number}> = ({model, descriptor, top = 120}) => {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{alignItems:'center',paddingTop:top,color:WHITE,textAlign:'center',pointerEvents:'none'}}>
    <Interactive.Div name={`${model} model stamp`} style={{fontFamily:DISPLAY_FONT,fontSize:126,lineHeight:.82,textTransform:'uppercase',textShadow:'0 4px 30px rgba(0,0,0,.65)',opacity:interpolate(frame,[3,10],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp'}),scale:interpolate(frame,[3,16],[1.35,1],{easing:Easing.bezier(.16,1,.3,1),extrapolateLeft:'clamp',extrapolateRight:'clamp'})}}>{model}</Interactive.Div>
    <Interactive.Div name={`${model} kinetic descriptor`} style={{fontFamily:DISPLAY_FONT,fontSize:45,lineHeight:1,color:INK,backgroundColor:ACCENT,padding:'10px 18px 13px',marginTop:24,letterSpacing:2,textTransform:'uppercase',opacity:interpolate(frame,[16,23],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp'}),translate:interpolate(frame,[16,28],['-100px 0px','0px 0px'],{easing:Easing.bezier(.16,1,.3,1),extrapolateLeft:'clamp',extrapolateRight:'clamp'})}}>{descriptor}</Interactive.Div>
  </AbsoluteFill>;
};

const Panel: React.FC<{children:React.ReactNode}> = ({children}) => <div style={{width:'100%',height:'33.333%',flexShrink:0,overflow:'hidden',position:'relative',borderBottom:`5px solid ${INK}`}}>{children}<div style={{position:'absolute',left:0,right:0,bottom:0,height:22,background:'linear-gradient(transparent,rgba(0,0,0,.68))'}}/></div>;
const TriplePanel: React.FC<{top: React.ReactNode; middle: React.ReactNode; bottom: React.ReactNode}> = ({top,middle,bottom}) => <AbsoluteFill style={{backgroundColor:INK}}><Panel>{top}</Panel><Panel>{middle}</Panel><Panel>{bottom}</Panel></AbsoluteFill>;

const Intro: React.FC = () => <AbsoluteFill><Footage src="emas7/front-reveal.mp4" trimBefore={8} name="Opening e.MAS 7 detail" position="48% 50%" dark={.62}/><KineticWords beats={[{text:'First car?'},{text:'Start electric.',accent:true}]} top={150} size={122}/></AbsoluteFill>;
const FivePanels: React.FC = () => <TriplePanel top={<Footage src="emas5-video/freedom.mp4" trimBefore={2360} name="e.MAS 5 infotainment" position="50% 50%" />} middle={<Footage src="emas5-video/freedom.mp4" trimBefore={2840} name="e.MAS 5 rolling hero" position="50% 50%" />} bottom={<Footage src="emas5-video/freedom.mp4" trimBefore={1015} name="e.MAS 5 rear detail" position="50% 50%" />} />;
const SevenPanels: React.FC = () => <TriplePanel top={<PortraitAsLandscape src="emas7-requested/top-115-optimized.mp4" trimBefore={24} name="e.MAS 7 cockpit - top clip 115" />} middle={<PortraitAsLandscape src="emas7-requested/center-25-optimized.mp4" trimBefore={24} name="e.MAS 7 front - centre clip 25" />} bottom={<PortraitAsLandscape src="emas7-requested/bottom-24-optimized.mp4" trimBefore={24} name="e.MAS 7 cargo - bottom clip 24" />} />;
const Lifestyle: React.FC = () => <AbsoluteFill><Footage src="emas5-video/freedom.mp4" trimBefore={3135} name="e.MAS 5 city drive" position="50% 50%" dark={.55}/><KineticWords beats={[{text:'City life?'},{text:'Made easy.',accent:true}]} top={145} size={108}/></AbsoluteFill>;
const SevenFeature: React.FC = () => <AbsoluteFill style={{backgroundColor:INK}}><div style={{position:'absolute',left:0,right:0,top:610,height:700,overflow:'hidden'}}><PortraitAsLandscape src="emas7/charging.mp4" trimBefore={5} name="e.MAS 7 charging landscape feature"/></div><KineticWords beats={[{text:'More room.'},{text:'More possibilities.',outline:true}]} top={125} size={89}/></AbsoluteFill>;
const Choice: React.FC = () => <AbsoluteFill style={{backgroundColor:INK}}><div style={{height:'50%',position:'relative',overflow:'hidden'}}><Footage src="emas5-video/freedom.mp4" trimBefore={1050} name="e.MAS 5 hero choice" position="50% 50%" dark={.28}/></div><div style={{height:'50%',position:'relative',overflow:'hidden',borderTop:`5px solid ${INK}`}}><Footage src="emas7-landscape/power-up.mp4" trimBefore={60} name="e.MAS 7 moving Singapore driving montage" position="50% 50%" dark={.3}/></div><AbsoluteFill style={{justifyContent:'center'}}><KineticWords beats={[{text:'Go compact?'},{text:'Or go big?',accent:true}]} top={720} size={91}/></AbsoluteFill></AbsoluteFill>;
const Cta: React.FC = () => <AbsoluteFill style={{backgroundColor:'black'}}><Video src={staticFile('cta/proton-logo.mp4')} muted style={{width:'100%',height:'100%',objectFit:'cover'}}/><AbsoluteFill style={{height:680,backgroundColor:'black'}}><KineticWords beats={[{text:'Your first EV.'},{text:'Your call.',accent:true}]} top={125} size={86}/><div style={{position:'absolute',top:510,width:'100%',fontFamily:BODY_FONT,fontSize:25,letterSpacing:4,color:WHITE,textAlign:'center'}}>e.MAS 5  ·  e.MAS 7</div></AbsoluteFill></AbsoluteFill>;

export const FirstCarVideo: React.FC = () => <AbsoluteFill style={{backgroundColor:INK}}>
  <FontStyles/>
  <Sequence durationInFrames={125}><Intro/></Sequence>
  <Sequence from={125} durationInFrames={145}><FivePanels/><ModelStamp model="e.MAS 5" descriptor="Compact. Quick. City-ready."/></Sequence>
  <Sequence from={270} durationInFrames={120}><SevenPanels/><ModelStamp model="e.MAS 7" descriptor="More space. More comfort."/></Sequence>
  <Sequence from={390} durationInFrames={120}><Lifestyle/></Sequence>
  <Sequence from={510} durationInFrames={100}><SevenFeature/></Sequence>
  <Sequence from={610} durationInFrames={150}><Choice/></Sequence>
  <Sequence from={760} durationInFrames={140}><Cta/></Sequence>
  <Audio src={staticFile('audio/music.wav')} volume={.22}/>
</AbsoluteFill>;
