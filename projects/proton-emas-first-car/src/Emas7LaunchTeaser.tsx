import React from 'react';
import {AbsoluteFill, Easing, interpolate, Sequence, staticFile, useCurrentFrame} from 'remotion';
import {Audio, Video} from '@remotion/media';

const CYAN = '#66F7FF';
const BLUE = '#174CFF';
const RED = '#FF285C';

type ShotProps = {
  src: string;
  name: string;
  trimBefore?: number;
  position?: string;
  zoom?: number;
  dark?: number;
  cool?: number;
};

const TechShot: React.FC<ShotProps> = ({src, name, trimBefore = 0, position = '50% 50%', zoom = 1.12, dark = .32, cool = .3}) => {
  const frame = useCurrentFrame();
  const glitch = frame % 37 < 3;
  const flash = frame < 4 ? interpolate(frame, [0, 3], [.7, 0], {extrapolateRight:'clamp'}) : 0;
  return <AbsoluteFill style={{backgroundColor:'#02050A',overflow:'hidden'}}>
    <Video name={name} src={staticFile(src)} trimBefore={trimBefore} muted style={{width:'100%',height:'100%',objectFit:'cover',objectPosition:position,scale:interpolate(frame,[0,80],[zoom,zoom+0.06],{extrapolateLeft:'clamp',extrapolateRight:'clamp'}),filter:`contrast(1.24) saturate(${1 + cool}) brightness(${1-dark}) hue-rotate(-7deg)`}}/>
    {glitch ? <>
      <Video src={staticFile(src)} trimBefore={trimBefore} muted style={{position:'absolute',inset:0,width:'100%',height:'100%',objectFit:'cover',objectPosition:position,translate:'-13px 0px',mixBlendMode:'screen',opacity:.32,filter:'sepia(1) saturate(8) hue-rotate(145deg)',clipPath:'inset(18% 0 61% 0)'}}/>
      <Video src={staticFile(src)} trimBefore={trimBefore} muted style={{position:'absolute',inset:0,width:'100%',height:'100%',objectFit:'cover',objectPosition:position,translate:'15px 0px',mixBlendMode:'screen',opacity:.28,filter:'sepia(1) saturate(8) hue-rotate(295deg)',clipPath:'inset(65% 0 12% 0)'}}/>
    </> : null}
    <AbsoluteFill style={{background:`linear-gradient(112deg,rgba(0,0,0,${dark+.18}) 0%,rgba(9,48,115,${cool*.24}) 42%,transparent 65%),radial-gradient(circle at 50% 58%,transparent 20%,rgba(0,0,0,.48) 100%)`,mixBlendMode:'multiply'}}/>
    <AbsoluteFill style={{backgroundColor:`rgba(13,65,190,${cool*.26})`,mixBlendMode:'color'}}/>
    <AbsoluteFill style={{opacity:.12,background:'repeating-linear-gradient(180deg,transparent 0px,transparent 5px,rgba(125,225,255,.24) 6px)'}}/>
    <AbsoluteFill style={{opacity:flash,backgroundColor:CYAN,mixBlendMode:'screen'}}/>
  </AbsoluteFill>;
};

const BlackPulse: React.FC = () => {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{backgroundColor:'#000',opacity:interpolate(frame,[0,2,5,8],[1,.1,.92,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp'})}}/>;
};

const LightSweep: React.FC = () => {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{pointerEvents:'none',overflow:'hidden'}}><div style={{position:'absolute',width:150,height:2400,top:-220,left:-300,rotate:'25deg',translate:interpolate(frame,[0,30],['0px 0px','1650px 0px'],{easing:Easing.bezier(.2,.8,.2,1),extrapolateLeft:'clamp',extrapolateRight:'clamp'}),background:`linear-gradient(90deg,transparent,${CYAN},white,transparent)`,filter:'blur(28px)',opacity:.42,mixBlendMode:'screen'}}/></AbsoluteFill>;
};

const EndLockup: React.FC = () => {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{backgroundColor:'#02050A',alignItems:'center',justifyContent:'center',color:'white',textAlign:'center'}}>
    <div style={{position:'absolute',width:720,height:720,borderRadius:'50%',background:`radial-gradient(circle,rgba(30,108,255,.25),transparent 67%)`,scale:interpolate(frame,[0,30],[.7,1.2],{extrapolateLeft:'clamp',extrapolateRight:'clamp'}),opacity:interpolate(frame,[0,10,48,60],[0,1,1,0],{extrapolateLeft:'clamp',extrapolateRight:'clamp'})}}/>
    <div style={{fontFamily:'Arial, Helvetica, sans-serif',fontSize:30,fontWeight:700,letterSpacing:11,color:CYAN,opacity:interpolate(frame,[5,16],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp'}),translate:interpolate(frame,[5,20],['0px 22px','0px 0px'],{easing:Easing.bezier(.16,1,.3,1),extrapolateLeft:'clamp',extrapolateRight:'clamp'})}}>PROTON</div>
    <div style={{fontFamily:'Arial, Helvetica, sans-serif',fontSize:112,lineHeight:.9,fontWeight:900,letterSpacing:-5,marginTop:24,opacity:interpolate(frame,[10,22],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp'}),scale:interpolate(frame,[10,28],[1.18,1],{easing:Easing.bezier(.16,1,.3,1),extrapolateLeft:'clamp',extrapolateRight:'clamp'})}}>e.MAS 7</div>
    <div style={{width:420,height:2,marginTop:34,background:`linear-gradient(90deg,transparent,${CYAN},transparent)`,scale:interpolate(frame,[18,34],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp'})}}/>
    <div style={{fontFamily:'Arial, Helvetica, sans-serif',fontSize:25,fontWeight:500,letterSpacing:7,marginTop:28,color:'#B9C7D6',opacity:interpolate(frame,[26,40],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp'})}}>ELECTRIC. ELEVATED.</div>
  </AbsoluteFill>;
};

export const Emas7LaunchTeaser: React.FC = () => <AbsoluteFill style={{backgroundColor:'#000'}}>
  <Sequence durationInFrames={42}><TechShot src="emas7-teaser/clip-26.mp4" name="Shadowed front hero" trimBefore={18} position="50% 52%" zoom={1.34} dark={.57}/><LightSweep/></Sequence>
  <Sequence from={42} durationInFrames={12}><BlackPulse/></Sequence>
  <Sequence from={54} durationInFrames={42}><TechShot src="emas7-teaser/clip-31.mp4" name="Headlamp macro" trimBefore={28} position="43% 48%" zoom={1.42} dark={.28}/></Sequence>
  <Sequence from={96} durationInFrames={34}><TechShot src="emas7-teaser/clip-29.mp4" name="Proton emblem detail" trimBefore={22} position="50% 55%" zoom={1.38} dark={.34}/></Sequence>
  <Sequence from={130} durationInFrames={38}><TechShot src="emas7-teaser/clip-80.mp4" name="Rear light detail" trimBefore={20} position="58% 50%" zoom={1.34} dark={.28} cool={.15}/></Sequence>
  <Sequence from={168} durationInFrames={10}><BlackPulse/></Sequence>
  <Sequence from={178} durationInFrames={44}><TechShot src="emas7-teaser/clip-47.mp4" name="Charging port reveal" trimBefore={22} position="50% 53%" zoom={1.38} dark={.32}/><LightSweep/></Sequence>
  <Sequence from={222} durationInFrames={46}><TechShot src="emas7-teaser/clip-93.mp4" name="Cockpit reveal" trimBefore={18} position="52% 50%" zoom={1.3} dark={.38}/></Sequence>
  <Sequence from={268} durationInFrames={44}><TechShot src="emas7-teaser/clip-106.mp4" name="Front seat detail" trimBefore={18} position="51% 50%" zoom={1.34} dark={.4}/></Sequence>
  <Sequence from={312} durationInFrames={44}><TechShot src="emas7-teaser/clip-75.mp4" name="Rear light bar hero" trimBefore={20} position="50% 48%" zoom={1.3} dark={.27} cool={.18}/></Sequence>
  <Sequence from={356} durationInFrames={42}><TechShot src="emas7-teaser/clip-63.mp4" name="Full exterior hero" trimBefore={20} position="50% 52%" zoom={1.3} dark={.3}/><LightSweep/></Sequence>
  <Sequence from={398} durationInFrames={52}><EndLockup/></Sequence>
  <Audio src={staticFile('audio/teaser-reference-track.mp3')} volume={.88}/>
</AbsoluteFill>;
