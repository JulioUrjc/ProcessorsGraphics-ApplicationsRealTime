uniform vec3 iResolution;
uniform float iGlobalTime;

const float NUMCELL = 15.0;

// Skeleton based in Inigo Quilez code 

const mat3 m = mat3( 0.00,  0.80,  0.60,
                    -0.80,  0.36, -0.48,
                    -0.60, -0.48,  0.64 );


vec3 Hash2(vec3 p)
{
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
    return vec3( minD2-minD, poligono,0);
}

vec3 env_landscape(float t, vec3 rd)
{
    vec3 light = normalize(vec3(sin(t), 0.6, cos(t)));
    float sun = max(0.0, dot(rd, light));
    float sky = max(0.0, dot(rd, vec3(0.0, 1.0, 0.0)));
    float ground = max(0.0, -dot(rd, vec3(0.0, 1.0, 0.0)));
    return 
        (pow(sun, 256.0)+0.2*pow(sun, 2.0))*vec3(2.0, 1.6, 1.0) +
        pow(ground, 0.5)*vec3(0.4, 0.3, 0.2) +
        pow(sky, 1.0)*vec3(0.5, 0.6, 0.7);
}

void main( void )
{
    vec2 p = (-iResolution.xy + 2.0*gl_FragCoord.xy) / iResolution.y;
    
    // camera movement  
    float an = 0.5*iGlobalTime;
    vec3 ro = vec3( 2.5*cos(an), 1.0, 2.5*sin(an) );
    vec3 ta = vec3( 0.0, 1.0, 0.0 );
    // camera matrix
    vec3 ww = normalize( ta - ro );
    vec3 uu = normalize( cross(ww,vec3(0.0,1.0,0.0) ) );
    vec3 vv = normalize( cross(uu,ww));
    // create view ray
    vec3 rd = normalize( p.x*uu + p.y*vv + 1.5*ww );
    // background
    //vec3 col = env_landscape(0.0, rd);
    vec3 col;
    // sphere center    
    vec3 sc = vec3(0.0,1.0,0.5);

    // raytrace-sphere
    float tmin = 10000.0;
    vec3  ce = ro - sc;
    float b = dot( rd, ce );
    float c = dot( ce, ce ) - .5;
    float h = b*b - c;

    vec3  nor = vec3(0.0);
    float occ = 1.0;
    vec3  pos = vec3(0.0);

    if( h>0.0 )
    {
        h = -b - sqrt(h);
        if( h<tmin ) 
        { 
            tmin=h; 
            // shading/lighting
            vec3 pos = ro + tmin*rd;
            vec3 texture = celular(pos*.5);
            //float f = celular( 1.0*pos ).x;
            //f *= occ;
            col = vec3(texture.x*1.2);
            col = mix( col, vec3(0.9), 1.0-exp( -0.003*tmin*tmin ) );
            //vec3 normal_s = pos-ce;
            //vec3 reflection = reflect(rd,normal_s);
            //col = vec3(texture.x);
        }
    }
 
    gl_FragColor = vec4( col, 1.0 );
}
