#define PI 3.14159265
#define TWOPI (2.0*PI)

/*
** Float blending modes
** Adapted from here: http://www.nathanm.com/photoshop-blending-math/
** But I modified the HardMix (wrong condition), Overlay, SoftLight, ColorDodge, ColorBurn, VividLight, PinLight (inverted layers) ones to have correct results
*/
#define BlendLinearDodgef 			BlendAddf
#define BlendLinearBurnf 			BlendSubtractf
#define BlendAddf(base, blend) 		min(base + blend, 1.0)
#define BlendSubtractf(base, blend) 	max(base + blend - 1.0, 0.0)
#define BlendLightenf(base, blend) 		max(blend, base)
#define BlendDarkenf(base, blend) 		min(blend, base)
#define BlendLinearLightf(base, blend) 	(blend < 0.5 ? BlendLinearBurnf(base, (2.0 * blend)) : BlendLinearDodgef(base, (2.0 * (blend - 0.5))))
#define BlendScreenf(base, blend) 		(1.0 - ((1.0 - base) * (1.0 - blend)))
#define BlendOverlayf(base, blend) 	(base < 0.5 ? (2.0 * base * blend) : (1.0 - 2.0 * (1.0 - base) * (1.0 - blend)))
#define BlendSoftLightf(base, blend) 	((blend < 0.5) ? (2.0 * base * blend + base * base * (1.0 - 2.0 * blend)) : (sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend)))
#define BlendColorDodgef(base, blend) 	((blend == 1.0) ? blend : min(base / (1.0 - blend), 1.0))
#define BlendColorBurnf(base, blend) 	((blend == 0.0) ? blend : max((1.0 - ((1.0 - base) / blend)), 0.0))
#define BlendVividLightf(base, blend) 	((blend < 0.5) ? BlendColorBurnf(base, (2.0 * blend)) : BlendColorDodgef(base, (2.0 * (blend - 0.5))))
#define BlendPinLightf(base, blend) 	((blend < 0.5) ? BlendDarkenf(base, (2.0 * blend)) : BlendLightenf(base, (2.0 *(blend - 0.5))))
#define BlendHardMixf(base, blend) 	((BlendVividLightf(base, blend) < 0.5) ? 0.0 : 1.0)
#define BlendReflectf(base, blend) 		((blend == 1.0) ? blend : min(base * base / (1.0 - blend), 1.0))

/*
** Vector3 blending modes
*/

// Component wise blending
#define Blend(base, blend, funcf) 		vec3(funcf(base.r, blend.r), funcf(base.g, blend.g), funcf(base.b, blend.b))

#define BlendNormal(base, blend) 		(blend)
#define BlendLighten				BlendLightenf
#define BlendDarken				BlendDarkenf
#define BlendMultiply(base, blend) 		(base * blend)
#define BlendAverage(base, blend) 		((base + blend) / 2.0)
#define BlendAdd(base, blend) 		min(base + blend, vec3(1.0))
#define BlendSubtract(base, blend) 	max(base + blend - vec3(1.0), vec3(0.0))
#define BlendDifference(base, blend) 	abs(base - blend)
#define BlendNegation(base, blend) 	(vec3(1.0) - abs(vec3(1.0) - base - blend))
#define BlendExclusion(base, blend) 	(base + blend - 2.0 * base * blend)
#define BlendScreen(base, blend) 		Blend(base, blend, BlendScreenf)
#define BlendOverlay(base, blend) 		Blend(base, blend, BlendOverlayf)
#define BlendSoftLight(base, blend) 	Blend(base, blend, BlendSoftLightf)
#define BlendHardLight(base, blend) 	BlendOverlay(blend, base)
#define BlendColorDodge(base, blend) 	Blend(base, blend, BlendColorDodgef)
#define BlendColorBurn(base, blend) 	Blend(base, blend, BlendColorBurnf)
#define BlendLinearDodge			BlendAdd
#define BlendLinearBurn			BlendSubtract
// Linear Light is another contrast-increasing mode
// If the blend color is darker than midgray, Linear Light darkens the image by decreasing the brightness. If the blend color is lighter than midgray, the result is a brighter image due to increased brightness.
#define BlendLinearLight(base, blend) 	Blend(base, blend, BlendLinearLightf)
#define BlendVividLight(base, blend) 	Blend(base, blend, BlendVividLightf)
#define BlendPinLight(base, blend) 		Blend(base, blend, BlendPinLightf)
#define BlendHardMix(base, blend) 		Blend(base, blend, BlendHardMixf)
#define BlendReflect(base, blend) 		Blend(base, blend, BlendReflectf)
#define BlendGlow(base, blend) 		BlendReflect(blend, base)
#define BlendPhoenix(base, blend) 		(min(base, blend) - max(base, blend) + vec3(1.0))
#define BlendOpacity(base, blend, F, O) 	(F(base, blend) * O + base * (1.0 - O))

#define NUM_BLENDMODES 25


int transform_sequence_ids[10];

float HueToRGB(float f1, float f2, float hue) {
	if (hue < 0.0)
		hue += 1.0;
	else if (hue > 1.0)
		hue -= 1.0;
	float res;
	if ((6.0 * hue) < 1.0)
		res = f1 + (f2 - f1) * 6.0 * hue;
	else if ((2.0 * hue) < 1.0)
		res = f2;
	else if ((3.0 * hue) < 2.0)
		res = f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
	else
		res = f1;
	return res;
}

vec3 HSLToRGB(vec3 hsl) {
	vec3 rgb;

	if (hsl.y == 0.0)
		rgb = vec3(hsl.z); // Luminance
	else
	{
		float f2;

		if (hsl.z < 0.5)
			f2 = hsl.z * (1.0 + hsl.y);
		else
			f2 = (hsl.z + hsl.y) - (hsl.y * hsl.z);

		float f1 = 2.0 * hsl.z - f2;

		rgb.r = HueToRGB(f1, f2, hsl.x + (1.0/3.0));
		rgb.g = HueToRGB(f1, f2, hsl.x);
		rgb.b= HueToRGB(f1, f2, hsl.x - (1.0/3.0));
	}

	return rgb;
}

/*
vec3 blend_any(vec3 base, vec3 blend, int mode) {
    if (mode==0) return BlendNormal(base, blend);
    else if(mode==1) return BlendLighten(base,blend);
    else if(mode==2) return BlendDarken(base,blend);
    else if(mode==3) return BlendMultiply(base, blend);
    else if(mode==4) return BlendAverage(base, blend);
    else if(mode==5) return BlendAdd(base, blend);
    else if(mode==6) return BlendSubtract(base, blend);
    else if(mode==7) return BlendDifference(base, blend);
    else if(mode==8) return BlendNegation(base, blend);
    else if(mode==9) return BlendExclusion(base, blend);
    else if(mode==10) return BlendScreen(base, blend);
    else if(mode==11) return BlendOverlay(base, blend);
    else if(mode==12) return BlendSoftLight(base, blend);
    else if(mode==13) return BlendHardLight(base, blend);
    else if(mode==14) return BlendColorDodge(base, blend);
    else if(mode==15) return BlendColorBurn(base, blend);
    else if(mode==16) return BlendLinearDodge(base, blend);
    else if(mode==17) return BlendLinearBurn(base, blend);
    else if(mode==18) return BlendLinearLight(base, blend);
    else if(mode==19) return BlendVividLight(base, blend);
    else if(mode==20) return BlendPinLight(base, blend);
    else if(mode==21) return BlendHardMix(base, blend);
    else if(mode==22) return BlendReflect(base, blend);
    else if(mode==23) return BlendGlow(base, blend);
    else if(mode==24) return BlendPhoenix(base, blend);
    //else if(mode==25) return BlendOpacity(base, blend, F, O);
    else return vec3(1,0,0);
}
*/

vec3 blend_any_opacity(vec3 base, vec3 blend, int mode, float opacity) {
    if (mode==0) return BlendOpacity(base, blend, BlendNormal, opacity);
    else if(mode==1) return BlendOpacity(base,blend, BlendLighten, opacity);
    else if(mode==2) return BlendOpacity(base,blend, BlendDarken, opacity);
    else if(mode==3) return BlendOpacity(base, blend, BlendMultiply, opacity);
    else if(mode==4) return BlendOpacity(base, blend, BlendAverage, opacity);
    else if(mode==5) return BlendOpacity(base, blend, BlendAdd, opacity);
    else if(mode==6) return BlendOpacity(base, blend, BlendSubtract, opacity);
    else if(mode==7) return BlendOpacity(base, blend, BlendDifference, opacity);
    else if(mode==8) return BlendOpacity(base, blend, BlendNegation, opacity);
    else if(mode==9) return BlendOpacity(base, blend, BlendExclusion, opacity);
    else if(mode==10) return BlendOpacity(base, blend, BlendScreen, opacity);
    else if(mode==11) return BlendOpacity(base, blend, BlendOverlay, opacity);
    else if(mode==12) return BlendOpacity(base, blend, BlendSoftLight, opacity);
    else if(mode==13) return BlendOpacity(base, blend, BlendHardLight, opacity);
    else if(mode==14) return BlendOpacity(base, blend, BlendColorDodge, opacity);
    else if(mode==15) return BlendOpacity(base, blend, BlendColorBurn, opacity);
    else if(mode==16) return BlendOpacity(base, blend, BlendLinearDodge, opacity);
    else if(mode==17) return BlendOpacity(base, blend, BlendLinearBurn, opacity);
    else if(mode==18) return BlendOpacity(base, blend, BlendLinearLight, opacity);
    else if(mode==19) return BlendOpacity(base, blend, BlendVividLight, opacity);
    else if(mode==20) return BlendOpacity(base, blend, BlendPinLight, opacity);
    else if(mode==21) return BlendOpacity(base, blend, BlendHardMix, opacity);
    else if(mode==22) return BlendOpacity(base, blend, BlendReflect, opacity);
    else if(mode==23) return BlendOpacity(base, blend, BlendGlow, opacity);
    else if(mode==24) return BlendOpacity(base, blend, BlendPhoenix, opacity);
    else return vec3(1,0,0);
}

vec3 lerp_blendmode(vec3 base, vec3 blend, float opacity1, float opacity2, int mode1, int mode2, float t) {
    vec3 b1 = blend_any_opacity(base, blend, mode1, opacity1);
    vec3 b2 = blend_any_opacity(base, blend, mode2, opacity2);
    return mix(b1, b2, fract(t));
}

vec3 blend_lerper(vec3 c1, vec3 c2, float o1, float o2, int blendmode_offset, float t) {
    int m1 = (blendmode_offset+int(t)) % NUM_BLENDMODES;
    int m2 = (m1+1) % NUM_BLENDMODES;
    return lerp_blendmode(c1, c2, o1, o2, m1, m2, t);
}

uniform float FrameCount;
uniform float AnimateTime;
//uniform vec2 Position;
uniform vec2 MousePos;
uniform float Zoom;
uniform float MaxIterations;

uniform float LineWidth;
uniform float JitterRadius;

uniform float LayerBlendMode0; // should be int... lazy
uniform float LayerBlendMode1; // should be int... lazy
uniform float LayerBlendMode2;
uniform float LayerBlendMode3;
uniform float LayerBlendMode4;
uniform float LayerBlendMode5;
uniform float LayerBlendMode6;
uniform float LayerBlendMode7;
uniform float LayerOpacity0;
uniform float LayerOpacity1;
uniform float LayerOpacity2;
uniform float LayerOpacity3;
uniform float LayerOpacity4;
uniform float LayerOpacity5;
uniform float LayerOpacity6;
uniform float LayerOpacity7;
uniform float BlendmodeInterpolator;
uniform float BlendmodeLerpSpeed;

uniform float TransformSqrt;
uniform float TransformLogOfZ;
uniform float TransformSqrtOfZ;
uniform float TransformInverse; // bool
uniform float TransformMobius; // bool
uniform float TransformZPlusOneOverZ;
uniform float TransformSinOfZ;
uniform float TransformTanOfZ;
uniform float TransformZSquaredPlusC;
uniform float TransformSinOfOneOverZ;

uniform float TransformInterpolator;
uniform float TransformLerpSpeed;

uniform float NumTransformIterations;
// uniform float TransformLerpTime;


uniform vec4 BackgroundColor;
uniform float LayerGridVisible;
uniform float LayerMandelbrotVisible;
uniform float LayerIntCirclesVisible;
uniform float LayerHSBMapVisible;
uniform float LayerImageVisible;
uniform float LayerCheckerVisible;
uniform float LayerCoordDotsVisible;
uniform float LayerNoiseVisible;
uniform float HSBMapAngleOffset;
uniform float TextureRepeat;
uniform sampler2D CurrentTexture;



float cosine_lerp(float y1,float y2, float mu) {
   float mu2 = (1.0-cos(mu*PI))* 0.5;
   return(y1*(1.0-mu2)+y2*mu2);
}

/**
 *
 * Complex math functions
 *
        ((a.x*a.x)+(a.y*a.y))
 */


#define cx_mul(a, b) vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x)
//#define cx_div(a, b) (cx_mul(a , cx_inv(b)))
#define cx_div(a, b) vec2(((a.x*b.x+a.y*b.y)/(b.x*b.x+b.y*b.y)),((a.y*b.x-a.x*b.y)/(b.x*b.x+b.y*b.y)))
#define cx_modulus(a) length(a)
#define cx_conj(a) vec2(a.x,-a.y)
#define cx_arg(a) atan2(a.y,a.x)
#define cx_sin(a) vec2(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y))
#define cx_cos(a) vec2(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y))
//#define cx_tan cx_div(cx_sin(a), cx_cos(a))

//vec2 cx_add(vec2 a, vec2 b) { return a+b;} // is redundant
//vec2 cx_sub(vec2 a, vec2 b) { return a-b;} // is redundant
//vec2 cx_mul(vec2 a, vec2 b) { return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);}
//vec2 cx_div(vec2 a, vec2 b) { return cx_mul(a , cx_inv(b));}
//float cx_modulus(vec2 a) {return sqrt(a.x*a.x+a.y*a.y);}
//float cx_modulus(vec2 a) {return length(a); }
//float cx_arg(vec2 a) {return atan2(a.y,a.x);}

vec2 cx_sqrt(vec2 a) {
    float r = sqrt(a.x*a.x+a.y*a.y);
    float rpart = sqrt(0.5*(r+a.x));
    float ipart = sqrt(0.5*(r-a.x));
    if (a.y < 0.0) ipart = -ipart;
    return vec2(rpart,ipart);
}

//vec2 cx_sin(vec2 a) {return vec2(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y));}
//vec2 cx_cos(vec2 a) {return vec2(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y));}
vec2 cx_tan(vec2 a) {return cx_div(cx_sin(a), cx_cos(a)); }

vec2 cx_log(vec2 a) {
    float rpart = sqrt((a.x*a.x)+(a.y*a.y));
    float ipart = atan2(a.y,a.x);
    if (ipart > PI) ipart=ipart-(2.0*PI);
    return vec2(log(rpart),ipart);
}

vec2 cx_mobius(vec2 a) {
    vec2 c1 = a - vec2(1.0,0.0);
    vec2 c2 = a + vec2(1.0,0.0);
    return cx_div(c1, c2);
}
vec2 cx_z_plus_one_over_z(vec2 a) {
    return a + cx_div(vec2(1.0,0.0), a);
    //return cx_add(a, cx_div(vec2(1.0,0.0), a));
}
vec2 cx_z_squared_plus_c(vec2 z, vec2 c) {
    return cx_mul(z, z) + c;
}

vec2 cx_sin_of_one_over_z(vec2 z) {
    return cx_sin(cx_div(vec2(1.0,0.0), z));
}



#define TransformID_Identity        0
#define TransformID_Inverse         1
#define TransformID_Mobius          2
#define TransformID_ZPlusOneOverZ   3
#define TransformID_SinOfZ          4
#define TransformID_TanOfZ          5
#define TransformID_ZSquaredPlusC   6
#define TransformID_SinOfOneOverZ   7
#define TransformID_Sqrt            8
#define TransformID_Log             9
#define NUM_TRANSFORM_IDS   8

vec2 cx_transform_by_id(int txid, vec2 a) {
    if (txid == TransformID_Identity) return a;
    if (txid == TransformID_Inverse) return cx_inv(a);
    if (txid == TransformID_Mobius) return cx_mobius(a);
    if (txid == TransformID_Sqrt) return cx_sqrt(a);
    if (txid == TransformID_Log) return cx_log(a);
    if (txid == TransformID_ZPlusOneOverZ) return cx_z_plus_one_over_z(a);
    if (txid == TransformID_SinOfZ) return cx_sin(a);
    if (txid == TransformID_TanOfZ) return cx_tan(a);
    if (txid == TransformID_ZSquaredPlusC) return cx_z_squared_plus_c(a, MousePos);
    if (txid == TransformID_SinOfOneOverZ) return cx_sin_of_one_over_z(a);
}

vec2 cx_transform_lerper(vec2 a, float t) {
    int num_transforms = 9;
    vec2 a1,a2;

    int tx1 = int(t) % NUM_TRANSFORM_IDS;
    int tx2 = (tx1+1) % NUM_TRANSFORM_IDS;
    a1 = cx_transform_by_id(tx1, a);
    a2 = cx_transform_by_id(tx2, a);
    //return mix(a1, a2, fract(t));
    return vec2(cosine_lerp(a1.x,a2.x,fract(t)), cosine_lerp(a1.y,a2.y, fract(t)));

}



//vec2 jitter(vec2 c) { return c + noise2(c) * JitterRadius; }
// END Complex math functions

float rand(vec2 co){ return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);}

vec2 jitter(vec2 c) {
    float xoff = ((rand(vec2(c.x,c.y))*2.0) -1.0) * JitterRadius; // * Zoom;
    float yoff = ((rand(vec2(c.y,c.x))*2.0) -1.0) * JitterRadius; // * Zoom;
    //vec2 n = noise2(ctmp);
    return (c + vec2(xoff,yoff));
}



//
// Noise
/*
vec4 rand2(vec2 A,vec2 B,vec2 C,vec2 D){

        vec2 s = vec2 (12.9898,78.233);

        vec4 tmp = vec4( dot(A,s),dot(B,s),dot(C,s),dot(D,s));

        return fract(tan(tmp)  * 43758.5453);

        }



float noise(vec2 coord,float d){

        vec2 C[4];

        C[0] = floor( coord * d)/d ;

        C[1] = C[0] + vec2(1.0/d ,0.0  );

        C[2] = C[0] + vec2(1.0/d ,1.0/d);

        C[3] = C[0] + vec2(0.0   ,1.0/d);



        vec2 p = fract(coord * d);

        vec2 q = 1.0 - p;

        vec4 w = vec4(q.x * q.y, p.x * q.y, p.x * p.y, q.x * p.y);



        return dot(vec4(rand2(C[0],C[1],C[2],C[3])),w);

        }

*/

vec4 rand(vec2 A,vec2 B,vec2 C,vec2 D){
    vec2 s=vec2(12.9898,78.233);
    vec4 tmp=vec4(dot(A,s),dot(B,s),dot(C,s),dot(D,s));
    return fract(sin(tmp) * 43758.5453)* 2.0 - 1.0;
}



float noise(vec2 coord,float d){
    vec2 C[4];
    float d1 = 1.0/d;
    C[0]=floor(coord*d)*d1;
    C[1]=C[0]+vec2(d1,0.0);
    C[2]=C[0]+vec2(d1,d1);
    C[3]=C[0]+vec2(0.0,d1);
    vec2 p=fract(coord*d);
    vec2 q=1.0-p;
    vec4 w=vec4(q.x*q.y,p.x*q.y,p.x*p.y,q.x*p.y);
    return dot(vec4(rand(C[0],C[1],C[2],C[3])),w);
}

/*
vec4 get_noise_color(vec2 a) {
       float level= -1.0 -log2 (min(length(dFdx(a)),length(dFdy(a))));
        //-1.0 is a bias shift to avoid flickering
        level = min(level,16.0); //limit the level. Equalient to a 65536x65536 texture
        float n = 0.5;
        for(int i = 3; i< int(level);i++){
            n +=  0.12 * noise(a, exp2(float(i)));
        }
        n +=  0.12 * noise(gl_TexCoord[0].xy, exp2(floor(level)+1.0)) * fract(level);

        // add the last level multiplied with the fraction of the
        // level calculation for a smooth filtering
        return  max(0.0,sin (n* 12.0))* vec4(0.3,0.8,0.4,0.0) + vec4(0.3,0.3,0.5,0.0) * n;
}
*/


vec4 get_noise_color(vec2 a) {
    //float n = noise(a, 128.0);
    float level= -log2(min(length(dFdx(a)),length(dFdy(a))));
    level = min(level,13.0); //limit the level to avoid slowness
    float n = 0.5;
    for(int i = 3; i< int(level);i++){ //3 is the lowest noise level
        n +=  0.12 * noise(a, exp2(float(i)));
    }
    return vec4(n,n,n,n);
}


// End Noise
//


float get_smooth_branch_factor(float u, float v) {
    float fmult = 2.0; // frequency multiplyer
    u = clamp(u,0.0,1.0);
    v = clamp(v,0.0,1.0);
    float a = 0.5 + (0.5 * -cos(u*fmult * (2.0*PI)));
    float b = 0.5 + (0.5 * -cos((u*2.0*fmult)*(2.0*PI)));
    return (a+((b-a)*v)); // lerp (a,b,v)
  }

vec4 mandelbrot(vec2 z) {
    //float tmp = -float(TIME_FROM_INIT)*AnimSpeed;
    float   real  = z.x; // gl_TexCoord[0].s * Zoom + Position.x;
    float   imag  = z.y; //gl_TexCoord[0].t * Zoom + Position.y;
    float   Creal = real;   // Change this line...
    float   Cimag = imag;   // ...and this one to get a Julia set

    float r2 = 0.0;
    float iter;

    for (iter = 0.0; iter < MaxIterations && r2 < 16.0; ++iter) {
        float tempreal = real;
        real = (tempreal * tempreal) - (imag * imag) + Creal;
        imag = 2.0 * tempreal * imag + Cimag;
        r2   = (real * real) + (imag * imag);
    }

    if(r2 < 16.0) {
        return vec4(0.0,0.0,0.0,1.0);
    }


    float modulus = sqrt(r2); //sqrt (ReZ*ReZ + ImZ*ImZ);
    float mu = iter - (log (log (modulus)))/ log (2.0);

    mu += 0.5;
    float lo_boundary=0.0;
    float nearest_boundary = round(mu);
    if (nearest_boundary < mu) {
        lo_boundary = nearest_boundary;
    }
    else {
        lo_boundary = nearest_boundary - 1.0;
    }
    mu -= lo_boundary;

    float arg = atan2(imag,real);
    arg = ((arg + PI) / (2.0* PI));


    // smooth branch shading
    float sbf = get_smooth_branch_factor(arg, mu);
    //return vec4(sbf,sbf,sbf,1.0);
    return vec4(sbf,sbf,sbf,sbf);

    // Base the color on the number of iterations
/*
    vec3 color;
    if (r2 < 4.0)
        color = InnerColor;
    else
        color = mix(OuterColor1, OuterColor2, fract(iter * 0.05));

    return vec4 (clamp(color, 0.0, 1.0), 1.0);
*/
}





  vec4 get_grid_pixel_color(vec2 c) {
    int grid_pixel = 0;
    float bri=0.;
    int val = 0;
    vec4 linecolor = vec4(1.0, 1.0, 1.0, 1.0);
    vec4 bgcolor = vec4(0.0,0.0,0.0,0.0);
    vec2 nearest_int = vec2(abs(c.x - round(c.x)), abs(c.y - round(c.y)));

    if ((nearest_int.x < LineWidth) && (nearest_int.y < LineWidth)) { // line intersection
      bri = 2.0 - (nearest_int.x+nearest_int.y) / LineWidth; // better
      //linecolor.a = bri;
      return linecolor * bri;
      //return vec4(linecolor, bri);
      //return vec4(bri,bri,bri,bri);
    }
    else if ((nearest_int.x < LineWidth) || (nearest_int.y < LineWidth)) {
      if (nearest_int.x < LineWidth) {
            bri = ((1.-nearest_int.x/(LineWidth)));
            //linecolor.a = bri;
            return linecolor * bri;
            //return vec4(1.0,1.0,1.0,bri);
            //return vec4(bri,bri,bri,bri);
      }
      else if (nearest_int.y < LineWidth) {
        bri = ((1.-nearest_int.y/(LineWidth)));
        //linecolor.a = bri;
        return linecolor * bri;
        //return vec4(1.0,1.0,1.0,bri);
        //return vec4(bri,bri,bri,bri);
      }
    }
    else {
        linecolor = bgcolor;
    }
    return linecolor;
    //return vec4(1.0,0.0,0.0,0.0);
  }


vec4 get_coord_dot_color(vec2 c, vec2 coord, vec3 dotcolor) {
    float rad = 0.3;
    float dist = distance(c, coord);
    float a = (dist<rad)?(1.0-dist/rad):0.0;
    return vec4(dotcolor, a);
}

vec4 get_integer_circles_color(vec2 c) {
    vec4 pixel;
    float line_width = 0.15;
    float dnorm = length(c);
    float nearest_int = abs(dnorm-float(round(dnorm)));
    if (nearest_int < LineWidth) {
        float a =  1.0 - nearest_int/LineWidth;
        pixel = vec4(a,a,a,a);
    }
    else {
        pixel = vec4(0.0,0.0,0.0,0.0);
    }
    return pixel;
}

/*
vec3 get_hsbmap_color(vec2 c) {
    return HSLToRGB(vec3((cx_arg(c) + PI) / (PI*2.0), 1.0, 0.5));
}
*/

vec3 get_hsbmap_color(vec2 c) {
//    return HSLToRGB(vec3( (((cx_arg(c) + PI) / (PI*2.0)) + HSBMapAngleOffset)*0.5 , 1.0, 0.5));
    //return HSLToRGB(vec3(fract(cx_arg(c) + HSBMapAngleOffset), 1.0, 0.5));

//    float hue = (cx_arg(c) + PI) / (PI*2.0); // 0..1

    return HSLToRGB(vec3( fract(((cx_arg(c) + PI) / (PI*2.0)) + HSBMapAngleOffset), 1.0, 0.5));
}

vec4 get_checkerboard_color(vec2 c) {
    return ((int(floor(c.x))%2!=0) ^^ (int(ceil(c.y))%2 != 0))? vec4(0.5,0.5,0.5,1.0): vec4(0.0,0.0,0.0,0.0);
}

vec4 get_image_color(vec2 c) {
    return texture2D(CurrentTexture, (c - vec2(1.0,1.0)) / 2.0);
}




vec2 do_cx_transforms(vec2 a) {
    for (int i=0; i < int(NumTransformIterations); i++) {
        if (TransformMobius == 1.0) {
            //a = cx_mobius(a);
            a = cx_mobius(cx_div(MousePos, a));
            a = cx_mobius(cx_div(MousePos, a));
        }
        if (TransformSqrt == 1.0)           a = cx_sqrt(a);
        if (TransformInverse == 1.0)        a = cx_inv(a);
        if (TransformZPlusOneOverZ == 1.0)  a = cx_z_plus_one_over_z(a);
        if (TransformZSquaredPlusC == 1.0)  a = cx_z_squared_plus_c(a, MousePos);
        if (TransformSinOfZ == 1.0)         a = cx_sin(a);
        if (TransformTanOfZ == 1.0)         a = cx_tan(a);
        if (TransformSinOfOneOverZ == 1.0)  a = cx_sin_of_one_over_z(a);
        if (TransformLogOfZ == 1.0)         a = cx_log(a);
        if (TransformInterpolator == 1.0)   a = cx_transform_lerper(a, AnimateTime*TransformLerpSpeed);
    }
    return a;
}

float get_local_space_scale_factor(vec2 a) {
    vec2 c1 = do_cx_transforms(a);
    vec2 c2 = do_cx_transforms(a + (normalize(a)*0.1));
    //return log( distance(c1,c2)) * 0.1;
    //return 1.0 - (1.0 / (1.0 + distance(c1,c2)));
    return (1.0 / (1.0 + distance(c1,c2)));


}
float get_local_space_scale_factor2(vec2 a) {
    vec2 c1 = do_cx_transforms(a);
    vec2 c2 = do_cx_transforms(a + (normalize(a)*1.0));
    //return log( distance(c1,c2)) * 0.1;
    //return 1.0 - (1.0 / (1.0 + distance(c1,c2)));
    return (1.0 / (1.0 + distance(c1,c2)));


}
float get_local_space_scale_factor3(vec2 a1, vec2 a2) {
    float d = abs(length(a1) - length(a2));
    return (1.0 / (1.0 + d));
    //return (d>1.0)? 1.0: (1.0 / d);
}


void main(void) {
    vec2 c = gl_TexCoord[0].st;
    vec2 c_untransformed = c;
    //vec2 tmp = MousePos;
    //c = jitter(c);

    c = do_cx_transforms(c);

    c = jitter(c);
    // composite the pixel values
    vec4 color = BackgroundColor;
    float blendtime = AnimateTime * BlendmodeLerpSpeed;

    if (LayerHSBMapVisible == 1.0) {
        vec3 px_hsbmap = get_hsbmap_color(c);
        color = vec4(blend_any_opacity(color.rgb, px_hsbmap.rgb, int(LayerBlendMode0), LayerOpacity0), 1.0);
    }
    if (LayerImageVisible == 1.0) {
        vec4 px_image = get_image_color(c);
        if (TextureRepeat == 0.0) {
            if (c.x >= -1.0 && c.x <= 1.0 &&  c.y >= -1.0 && c.y <= 1.0) {
                color = vec4(blend_any_opacity(color.rgb, px_image.rgb, int(LayerBlendMode1), LayerOpacity1*px_image.a), 1.0);
            }
        }
        else {
            if (BlendmodeInterpolator == 1.0) {
                //color = vec4(blend_lerper(color.rgb, px_image.rgb, LayerOpacity1*px_image.a, LayerOpacity1*px_image.a, int(LayerBlendMode1), blendtime), 1.0);
                color = vec4(blend_lerper(color.rgb, px_image.rgb, LayerOpacity1, LayerOpacity1, int(LayerBlendMode1), blendtime), 1.0);
            } 
            else {
                 color = vec4(blend_any_opacity(color.rgb, px_image.rgb, int(LayerBlendMode1), LayerOpacity1*px_image.a), 1.0);
            }
        }
    }
    if (LayerCheckerVisible == 1.0) {
        vec4 px_checker = get_checkerboard_color(c);
        color = vec4(blend_any_opacity(color.rgb, px_checker.rgb, int(LayerBlendMode2), LayerOpacity2*px_checker.a), 1.0);
    }
    if (LayerGridVisible == 1.0) {
        vec4 px_grid = get_grid_pixel_color(c);
        //color = vec4(blend_any_opacity(color.rgb, px_grid.rgb, int(LayerBlendMode3), LayerOpacity3*px_grid.a), 1.0);
        if (BlendmodeInterpolator == 1.0) {
            color = vec4(blend_lerper(color.rgb, px_grid.rgb, LayerOpacity3, LayerOpacity3, int(LayerBlendMode3), blendtime), 1.0);
        }
        else {
            color = vec4(blend_any_opacity(color.rgb, px_grid.rgb, int(LayerBlendMode3), LayerOpacity3), 1.0);
        }
    }
    if (LayerIntCirclesVisible == 1.0) {
        vec4 px_intcircles = get_integer_circles_color(c);
        //color = vec4(blend_any_opacity(color.rgb, px_intcircles.rgb, int(LayerBlendMode4), LayerOpacity4*px_intcircles.a), 1.0);

        //color = vec4(blend_any_opacity(color.rgb, px_intcircles.rgb, int(LayerBlendMode4), LayerOpacity4), 1.0);
        if (BlendmodeInterpolator == 1.0) {
            color = vec4(blend_lerper(color.rgb, px_intcircles.rgb, LayerOpacity4, LayerOpacity4, int(LayerBlendMode4), blendtime), 1.0);
        }
        else {
            color = vec4(blend_any_opacity(color.rgb, px_intcircles.rgb, int(LayerBlendMode4), LayerOpacity4), 1.0);
        }
    }
    if (LayerMandelbrotVisible == 1.0) {
        vec4 px_mandel = mandelbrot(c);
        if (BlendmodeInterpolator == 1.0) {
            color = vec4(blend_lerper(color.rgb, px_mandel.rgb, LayerOpacity5, LayerOpacity5, int(LayerBlendMode5), blendtime), 1.0);
        }
        else {
            color = vec4(blend_any_opacity(color.rgb, px_mandel.rgb, int(LayerBlendMode5), LayerOpacity5*px_mandel.a), 1.0);
        }
    }
    if (LayerNoiseVisible == 1.0) {
        vec4 px_noise = get_noise_color(c);
        if (BlendmodeInterpolator == 1.0) {
            color = vec4(blend_lerper(color.rgb, px_noise.rgb, LayerOpacity7, LayerOpacity7, int(LayerBlendMode7), blendtime), 1.0);
        }
        else {
            color = vec4(blend_any_opacity(color.rgb, px_noise.rgb, int(LayerBlendMode7), LayerOpacity7), 1.0);
        }
    }




/*
    if (LayerCoordDotsVisible == 1.0) {
        vec4 px_coorddot = get_coord_dot_color(c, MousePos, vec3(1.0, 0.0, 0.0));
        color = vec4(blend_any_opacity(color.rgb, px_coorddot.rgb, int(LayerBlendMode6), LayerOpacity6*px_coorddot.a), 1.0);
        vec4 px_coorddot2 = get_coord_dot_color(c_untransformed, MousePos, vec3(0.0, 1.0, 0.0));
        color = vec4(blend_any_opacity(color.rgb, px_coorddot2.rgb, int(LayerBlendMode6), LayerOpacity6*px_coorddot2.a), 1.0);
    }
*/

    // test local space scaling metric
    if (LayerCoordDotsVisible == 1.0) {
        //float s1 = get_local_space_scale_factor(c_untransformed);
        float s1 = get_local_space_scale_factor3(c_untransformed, c);

        //float s1 = distance(c, c_untransformed);
        //float s1 = length(c - c_untransformed);
        //float s2 = (1.0 / (1.0 + s1));

        // float spscale = 1.0 / log(s1);
        //float spscale = log(s1);
        float spscale = s1; // clamp(s1, 0.0,1.0);
        
        //spscale = smoothstep(0.0, 1.0, spscale);

        // spscale = min(length(dFdx(c)),length(dFdy(c)));

        vec4 px_scalespace = vec4(spscale, spscale,spscale,spscale);
        //vec4 px_coorddot = get_coord_dot_color(c, MousePos, vec3(1.0, 0.0, 0.0));
        
        color = vec4(blend_any_opacity(color.rgb, px_scalespace.rgb, int(LayerBlendMode6), LayerOpacity6), 1.0);
        //color = vec4(blend_lerper(color.rgb, px_scalespace.rgb, LayerOpacity6, LayerOpacity6, int(LayerBlendMode6), TransformLerpTime), 1.0);

        //vec4 px_coorddot2 = get_coord_dot_color(c_untransformed, MousePos, vec3(0.0, 1.0, 0.0));
        //color = vec4(blend_any_opacity(color.rgb, px_coorddot2.rgb, int(LayerBlendMode6), LayerOpacity6*px_coorddot2.a), 1.0);
    }


    // implement wrapmode=background here
    // ...
    gl_FragColor = color;
}


