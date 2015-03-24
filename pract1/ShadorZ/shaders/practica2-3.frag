uniform vec3 iResolution;
uniform float iGlobalTime;

const float NUMCELL = 15.0;

// Skeleton based in Inigo Quilez code 

const mat3 m = mat3( 0.00,  0.80,  0.60,
                    -0.80,  0.36, -0.48,
                    -0.60, -0.48,  0.64 );

vec3 Hash2(vec3 p){
    float r = 523.0*sin(dot(p, vec2(53.3158, 43.6143)));
    return vec3(fract(15.32354 * r), fract(17.25865 * r), fract(19.32354 * r));
}

vec3 celular(vec3 p){
    p *= NUMCELL;
    float minD = 2.0;
    float minD2 = 2.0;
    float minAux=0.0;
    float poligono=0.0;
    vec3 entera = floor(p);

    for (int i = -1; i <= 1; i++){
        for (int j = -1; j <= 1; j++){
            for (int k = -1; k <= 1; k++){
                vec3 auxP = entera + vec3(i, j, k);
                vec3 hashP = Hash2(mod(auxP, NUMCELL));
                //vec2 hashP = Hash2(auxP);
            
                minD2 = min(minD2, length(p - auxP - hashP));   
                vec3 pos = vec3( float(i),float(j),float(k) );
                if (minD2<minD){
                    minAux= minD2;
                    minD2=minD;
                    minD=minAux;
                    poligono = Hash2(length(mod(entera+pos, NUMCELL)));
                    //poligono = Hash2(length(entera+pos));
                } 
            }
        }
    }
    return vec3( minD2-minD, poligono, 0.0f);
    //return vec3( minD, poligono, 0.0f);
}

vec3 env_landscape(float t, vec3 rd){
    vec3 light = normalize(vec3(sin(t), 0.6, cos(t)));
    float sun = max(0.0, dot(rd, light));
    float sky = max(0.0, dot(rd, vec3(0.0, 1.0, 0.0)));
    float ground = max(0.0, -dot(rd, vec3(0.0, 1.0, 0.0)));
    return 
        (pow(sun, 256.0)+0.2*pow(sun, 2.0))*vec3(2.0, 1.6, 1.0) +
        pow(ground, 0.5)*vec3(0.4, 0.3, 0.2) +
        pow(sky, 1.0)*vec3(0.5, 0.6, 0.7);
}

vec3 map( in vec3 p ){
    //p.y += 0.2*iGlobalTime;
    vec3 q = vec3( 4.0*fract(0.5+p.x/4.0)-2.0, p.y, 4.0*fract(0.5+p.z/4.0)-2.0 );
    

    float height = 0.9;

    vec3 v = celular(p*0.5);
    float f = clamp( 3.5*(v.y-v.x), 0.0, 0.5 );
    float d1 = length(q) - 1.0 + 0.2*f*height;
    
    return vec3(d1, mix(1.0,f,height), 0.0 );
}

vec3 calcNormal( in vec3 pos ){
    vec3 eps = vec3(0.001,0.0,0.0001);

    return normalize( vec3(
           map(pos+eps.xyy).x - map(pos-eps.xyy).x,
           map(pos+eps.yxy).x - map(pos-eps.yxy).x,
           map(pos+eps.yyx).x - map(pos-eps.yyx).x ) );
}

vec3 intersect( in vec3 ro, in vec3 rd, in float h )
{
    float t=0.0;
    float o;
    float m = -1;
    if (h>0.0)
    {
        vec3 res = map(ro+rd*t);
        return vec3(res.x,res.y,res.z);
        h = res.x;
        o = res.y;
        m = res.z;
    }
    return vec3(t,o,m);
    // float maxd = 1.0;
    // float precis = 0.001;
    // float h = precis*10.0;
    // float t = 0.0;
    // float m = -1.0;
    // float o = 0.0;
    // for( int i=0; i<20; i++ )
    // {
    //     if( h<precis||t>maxd ) break;
    //     t += h;
    //     vec3 res = map( ro+rd*t );
    //     //vec3 res;
    //     h = res.x;
    //     o = res.y;
    //     m = res.z;
    // }

    // if( t>maxd ) m=-1.0;
    // return vec3( t, o, m );
}

void main( void ){
    vec2 p = (-iResolution.xy + 2.0*gl_FragCoord.xy) / iResolution.y;
    
    // camera movement  
    float an = 0.5*iGlobalTime;
    //float an = 0.9;
    vec3 ro = vec3( 2.5*cos(an), 1.0, 2.5*sin(an) );
    vec3 ta = vec3( 0.0, 1.0, 0.0 );
    // camera matrix
    vec3 ww = normalize( ta - ro );
    vec3 uu = normalize( cross(ww,vec3(0.0,1.0,0.0) ) );
    vec3 vv = normalize( cross(uu,ww));
    // create view ray
    vec3 rd = normalize( p.x*uu + p.y*vv + 1.5*ww );
    // background
    vec3 col = env_landscape(0.0, rd);
    //vec3 col;
    // sphere center    
    vec3 sc = vec3(0.1,1.0,0.5);

    // raytrace-sphere
    float tmin = 3.0;
    vec3  ce = ro - sc;
    float b = dot( rd, ce );
    float c = dot( ce, ce ) - .5;
    float h = b*b - c;
    vec3 inter = intersect(ro,rd,h);
    //h -= inter.z;
    //h = inter.z;
    vec3  nor = vec3(0.0);
    float occ = 1.0;
    vec3  pos = vec3(0.0);

    if( h>-0.0){
        h = -b - sqrt(h);
        //if( h<tmin ) { 
            tmin=h; 
            // shading/lighting
            vec3 pos = ro + tmin*rd;
            vec3 texture = celular(pos*0.5);
            //float f = celular( 1.0*pos ).x;
            //f *= occ;
            //texture.x=texture.x;
            //col = vec3(texture.x, texture.x, texture.x);
           
            float brigth = 0.8;  // Brillo 
            float bordTam=1.9;   // Tamaño del borde
            float spotColor=0.8; // Como de plano es el color 0 plano o tesela

            col = brigth*sin(texture.y*NUMCELL+vec3(0.5,0.2,0.8))-spotColor*texture.x;
            col += bordTam*(2.0-smoothstep(0.0,0.12, texture.x)-smoothstep(0.0,0.04,texture.x));
            col *= mix(vec3(0.0,0.0,0.0), col, smoothstep(0.0, 0.0, texture.x));

            //vec3  pos = ro + tmat.x*rd;
            vec3  nor = calcNormal(pos);
            vec3 reflection = reflect(rd,nor);
            col += env_landscape(0.0, reflection*.5);    
            vec3 mate = vec3(0.7,0.57,0.5);     
            col *= mix( mate, (.2), smoothstep(0.7,1.0,nor.y));
            
            //col += vec3(texture.x*1.2);
            //col *= mix( col, vec3(0.9), 1.0-exp( -0.003*tmin*tmin ) );
            //vec3 normal_s = pos-ce;
            //vec3 reflection = reflect(rd,normal_s);
            //col = vec3(texture.x);
        //}
    }
 
    gl_FragColor = vec4( col, 1.0 );
}
