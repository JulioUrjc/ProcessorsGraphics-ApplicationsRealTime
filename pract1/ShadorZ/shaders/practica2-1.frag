uniform vec3 iResolution;
uniform float iGlobalTime;

const float NUMCELL = 15.0;

const int colorLookUp[256] = {4,3,1,1,1,2,4,2,2,2,5,1,0,2,1,2,2,0,4,3,2,1,2,1,3,2,2,4,2,2,5,1,2,3,2,2,2,2,2,3,2,4,2,5,3,2,2,2,5,3,3,5,2,1,3,3,4,4,2,3,0,4,2,2,2,1,3,2,2,2,3,3,3,1,2,0,2,1,1,2,2,2,2,5,3,2,3,2,3,2,2,1,0,2,1,1,2,1,2,2,1,3,4,2,2,2,5,4,2,4,2,2,5,4,3,2,2,5,4,3,3,3,5,2,2,2,2,2,3,1,1,4,2,1,3,3,4,3,2,4,3,3,3,4,5,1,4,2,4,3,1,2,3,5,3,2,1,3,1,3,3,3,2,3,1,5,5,4,2,2,4,1,3,4,1,5,3,3,5,3,4,3,2,2,1,1,1,1,1,2,4,5,4,5,4,2,1,5,1,1,2,3,3,3,2,5,2,3,3,2,0,2,1,1,4,2,1,3,2,1,2,2,3,2,5,5,3,4,5,5,2,4,4,5,3,2,2,2,1,4,2,3,3,4,2,5,4,2,4,2,2,2,4,5,3,2};
//lookUp table with poisson

vec2 Hash2(vec2 p){
	float auxTime = iGlobalTime/50000;
	auxTime = 0.0;
	float r = 523.0*sin(auxTime+dot(p, vec2(53.3158, 43.6143)));
	return vec2(fract(15.32354 * r), fract(17.25865 * r));
}

float minDistance(in vec2 p){
	p *= NUMCELL;
	float minD = 2.0;
	float minD2 = 2.0;
	float minD3 = 2.0;
	float minD4 = 2.0;
	float minAux=0;

	for (int i = -1; i <= 1; i++){
		for (int j = -1; j <= 1; j++){
			vec2 auxP = floor(p) + vec2(i, j);
			vec2 hashP = Hash2(mod(auxP, NUMCELL));
			//Euclidea distance
			//minD = min(minD, sqrt(pow((p.x-auxP.x-hashP.x),2)+pow((p.y-auxP.y-hashP.y),2)));
			//minD = min(minD, length(p - auxP - hashP)); //mod to do a circular texture
			//Manhattan distance
			//minD = min(minD, (abs(p.x-auxP.x-hashP.x)+abs(p.y-auxP.y-hashP.y)));
			////Teselas mode
			minD4 = min(minD4, length(p - auxP - hashP));
			//minD3 = min(minD3, (abs(p.x-auxP.x-hashP.x)+abs(p.y-auxP.y-hashP.y)));
			if (minD4<minD3){
				minAux= minD4;
				minD4=minD3;
				minD3=minAux;
			} 
			if (minD3<minD2){
				minAux= minD3;
				minD3=minD2;
				minD2=minAux;
			} 
			if (minD2<minD){
				minAux= minD2;
				minD2=minD;
				minD=minAux;
			} 
		}
	}
	float c1=.1;
	float c2=.1;
	float c3=.2;
	float c4=.6;
	//minD= pow(minD,2);
	//minD2= pow(minD2,2);
	//minD3= pow(minD3,2);
	//minD4= pow(minD4,2);

	//return minD;                      //Celular
	// Functions
	//return c1*minD2+c2*minD;          
	//return c3*minD3-c1*minD2-c2*minD;
	//return c4*minD4+c3*minD3+c2*minD2-c1*minD;
	//return minD*minD2;
	return minD2-minD;
	//return 2*minD3-minD2-minD;
}

void main(void){
	vec2 uv = gl_FragCoord.xy / iResolution.xy;
	float minD = minDistance(uv);
	//minD = 1.0 - minD;    // Inverso
	//vec3 color = vec3(minD*.93, minD*.23, minD*.13);
	//vec3 color = vec3(minD, minD,minD);
	//color*= vec3(uv.x+0.1,uv.y+0.1,uv.x+0.1);

	vec3 color = minD*vec3(0.0,0.0,10.0);
    color = mix(vec3(0.7,0.0,0.3), color, smoothstep(0.0, 0.0, minD));

	gl_FragColor = vec4(color ,0.8);
}