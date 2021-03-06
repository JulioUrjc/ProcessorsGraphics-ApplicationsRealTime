// Created by inigo quilez - iq/2013
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

uniform vec3 iResolution; 
uniform float iGlobalTime; 
uniform vec2 iMouse; 
uniform sampler2D iChannel3; 

float fbm( vec3 p, vec3 n )
{
	p *= 0.15;

	float x = texture2D( iChannel3, p.yz ).x;
	float y = texture2D( iChannel3, p.zx ).x;
	float z = texture2D( iChannel3, p.xy ).x;

	return x*abs(n.x) + y*abs(n.y) + z*abs(n.z);
}

float hash( float n )
{
    return fract(sin(n)*43758.5453);
}

vec3 hash3( float n )
{
    return fract(sin(vec3(n,n+1.0,n+2.0))*vec3(43758.5453123,22578.1459123,19642.3490423));
}

vec3 random3f( vec3 p )
{
#if 0
	return texture2D( iChannel0, (p.xy + vec2(3.0,1.0)*p.z+0.5)/256.0, -100.0 ).xyz;
#else
	vec3 n = vec3( dot(p,vec3(1.0,57.0,113.0)), 
				   dot(p,vec3(57.0,113.0,1.0)),
				   dot(p,vec3(113.0,1.0,57.0)));
	
    return fract(sin(n)*43758.5453);
#endif	
}
vec3 voronoi( in vec3 x )
{
    vec3 p = floor( x );
    vec3 f = fract( x );

	float id = 0.0;
    vec2 res = vec2( 100.0 );
    for( int k=-1; k<=1; k++ )
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec3 b = vec3( float(i), float(j), float(k) );
        vec3 r = vec3( b ) - f + random3f( p + b );
        float d = dot( r, r );

        if( d < res.x )
        {
			id = dot( p+b, vec3(1.0,57.0,113.0 ) );
            res = vec2( d, res.x );			
        }
        else if( d < res.y )
        {
            res.y = d;
        }
    }

    return vec3( sqrt( res ), abs(id) );
}

vec3 map( in vec3 p ){
	//p.y += 0.2*iGlobalTime;
	vec3 q = vec3( 4.0*fract(0.5+p.x/4.0)-2.0, p.y, 4.0*fract(0.5+p.z/4.0)-2.0 );
	

	float height = 0.9;

	vec3 v = voronoi(2.0*p);
	float f = clamp( 3.5*(v.y-v.x), 0.0, 0.5 );
	float d1 = length(q) - 1.0- 0.2*f*height;
	
    return vec3(d1, mix(1.0,f,height), 0.0 );
}

float map2( in vec3 p )
{
	vec3 q = vec3( 4.0*fract(0.5+p.x/4.0)-2.0, p.y, 4.0*fract(0.5+p.z/4.0)-2.0 );
	vec2 id = floor( 0.5+p.xz/4.0 );
    q.xz += 0.5*(-1.0+2.0*vec2(hash(id.x+113.0*id.y),hash(13.0*id.x+57.0*id.y)));
	q.y -= 0.5;

	return length(q.xz) - 1.1;

}

vec3 intersect( in vec3 ro, in vec3 rd )
{
	float maxd = 5.0;
	float precis = 0.001;
    float h = precis*1.0;
    float t = 0.0;
    float m = -1.0;
	float o = 0.0;
    for( int i=0; i<50; i++ )
    {
        if( h<precis||t>maxd ) break;
        t += h;
	    vec3 res = map( ro+rd*t );
	    //vec3 res;
        h = res.x;
		o = res.y;
	    m = res.z;
    }

    if( t>maxd ) m=-1.0;
    return vec3( t, o, m );
}

vec3 calcNormal( in vec3 pos ){
    vec3 eps = vec3(0.0001,0.0,0.0);

	return normalize( vec3(
           map(pos+eps.xyy).x - map(pos-eps.xyy).x,
           map(pos+eps.yxy).x - map(pos-eps.yxy).x,
           map(pos+eps.yyx).x - map(pos-eps.yyx).x ) );
}

float calcCurvature( in vec3 pos, in vec3 nor )
{
	float totao = 0.0;
    for( int aoi=0; aoi<4; aoi++ )
    {
		vec3 aopos = normalize(hash3(float(aoi)*213.47));
		aopos = aopos - dot(nor,aopos)*nor;
		aopos = pos + aopos*0.05;
        float dd = clamp( map( aopos ).x*300.0, 0.0, 1.0 );
        totao += dd;
    }
	
    return smoothstep( 0.2, 1.0, totao/4.0 );
}

vec3 doBumpMap( in vec3 pos, in vec3 nor, float amount )
{
    float e = 0.001;
    float b = 0.01;
    
	float ref = fbm( 48.0*pos, nor );
    vec3 gra = -b*vec3( fbm(48.0*vec3(pos.x+e, pos.y, pos.z),nor)-ref,
                        fbm(48.0*vec3(pos.x, pos.y+e, pos.z),nor)-ref,
                        fbm(48.0*vec3(pos.x, pos.y, pos.z+e),nor)-ref )/e;

	amount *= 0.2 + 0.8*smoothstep( 0.3, 0.45, fbm(4.0*pos,nor) );

	vec3 tgrad = gra - nor * dot ( nor , gra );
    return normalize ( nor - amount*tgrad );
}

void main(void)
{
	vec2 q = gl_FragCoord.xy / iResolution.xy;
    vec2 p = -1.0 + 2.0 * q;
    p.x *= iResolution.x/iResolution.y;
    vec2 m = iMouse.xy/iResolution.xy;

	
    // camera
    float r2 = p.x*p.x*0.32 + p.y*p.y;
    p *= 0.5 + 0.5*(7.0-sqrt(37.5-11.5*r2))/(r2+1.0);
    //float an = 0.04*iGlobalTime - 7.0*m.x;
    //vec3 ro = 3.1*normalize(vec3(sin(an),0.1+0.6-0.4*m.y, cos(an)));
    //float an = 0.02:
    vec3 ro = 3.1*normalize(vec3(0.5,0.1+0.6, 0.5));
    vec3 ta = vec3( 0.0, 0.75, 0.0 );
    vec3 ww = normalize( ta - ro );
    vec3 uu = normalize( cross(ww,vec3(0.5,0.5,0.0) ) );
    vec3 vv = normalize( cross(uu,ww));
    vec3 rd = normalize( p.x*uu + p.y*vv + 1.5*ww );

	// render
    vec3 col = vec3(0.7);

	// raymarch
    vec3 tmat = intersect(ro,rd);
    if( tmat.z>-0.5 )
    {
        vec3  pos = ro + tmat.x*rd;
        vec3  nor = calcNormal(pos);	
        //vec3 nor;
        col = vec3(0.0,1.0,0.0);
		//vec3 mate = vec3(0.7,0.57,0.5);		
		col = mix( vec3(0.5), (.2), smoothstep(0.7,1.0,nor.y));
	}

	
    gl_FragColor = vec4( col,1.0 );
}