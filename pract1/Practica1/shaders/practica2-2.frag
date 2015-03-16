uniform vec3 iResolution;
uniform float iGlobalTime;

vec2 Hash2(vec2 p)
{
    float r = 523.0*sin(dot(p, vec2(53.3158, 43.6143)));
    return vec2(fract(15.32354 * r), fract(17.25865 * r));
}

void main(void)
{
    gl_FragColor = vec4(1.0, 1.0, 0.0 ,1.0);
}