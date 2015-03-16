uniform vec3 iResolution;
uniform float iGlobalTime;

// Skeleton based in Inigo Quilez code 

const mat3 m = mat3( 0.00,  0.80,  0.60,
                    -0.80,  0.36, -0.48,
                    -0.60, -0.48,  0.64 );


vec2 Hash2(vec2 p)
{
    float r = 523.0*sin(dot(p, vec2(53.3158, 43.6143)));
    return vec2(fract(15.32354 * r), fract(17.25865 * r));
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
    vec3 col = env_landscape(0.0, rd);
    // sphere center    
    vec3 sc = vec3(0.0,1.0,0.5);

    // raytrace-sphere
    float tmin = 10000.0;
    vec3  ce = ro - sc;
    float b = dot( rd, ce );
    float c = dot( ce, ce ) - .5;
    float h = b*b - c;
    if( h>0.0 )
    {
        h = -b - sqrt(h);
        if( h<tmin ) 
        { 
            tmin=h; 
            // shading/lighting
            vec3 pos = ro + tmin*rd;
            vec3 normal_s = pos-ce;
            vec3 reflection = reflect(rd,normal_s);
            col = env_landscape(0.0, reflection*.5);
        }
    }
 
    gl_FragColor = vec4( col, 1.0 );
}
