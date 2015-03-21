uniform vec3 iResolution;
uniform float iGlobalTime;

const float NUMCELL = 20.0;

const int colorLookUp[256] = {4,3,1,1,1,2,4,2,2,2,5,1,0,2,1,2,2,0,4,3,2,1,2,1,3,2,2,4,2,2,5,1,2,3,2,2,2,2,2,3,2,4,2,5,3,2,2,2,5,3,3,5,2,1,3,3,4,4,2,3,0,4,2,2,2,1,3,2,2,2,3,3,3,1,2,0,2,1,1,2,2,2,2,5,3,2,3,2,3,2,2,1,0,2,1,1,2,1,2,2,1,3,4,2,2,2,5,4,2,4,2,2,5,4,3,2,2,5,4,3,3,3,5,2,2,2,2,2,3,1,1,4,2,1,3,3,4,3,2,4,3,3,3,4,5,1,4,2,4,3,1,2,3,5,3,2,1,3,1,3,3,3,2,3,1,5,5,4,2,2,4,1,3,4,1,5,3,3,5,3,4,3,2,2,1,1,1,1,1,2,4,5,4,5,4,2,1,5,1,1,2,3,3,3,2,5,2,3,3,2,0,2,1,1,4,2,1,3,2,1,2,2,3,2,5,5,3,4,5,5,2,4,4,5,3,2,2,2,1,4,2,3,3,4,2,5,4,2,4,2,2,2,4,5,3,2};
//lookUp table with poisson

vec2 Hash2(vec2 p){
	float auxTime = iGlobalTime/50000;
	//auxTime = 0.0;
	float r = 523.0*sin(auxTime+dot(p, vec2(53.3158, 43.6143)));
	return vec2(fract(15.32354 * r), fract(17.25865 * r));
}

float minDistance(in vec2 p){
	p *= NUMCELL;
	float minD = 2.0;
	float minD2 = 2.0;
	float minD3 = 2.0;
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
			minD3 = min(minD3, length(p - auxP - hashP));
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
	//return minD;               //Celular
	return minD2-minD;         //Teselas
	//return minD3-minD2-minD;   //Teselas Distantes
}

void main(void){
	vec2 uv = gl_FragCoord.xy / iResolution.xy;
	float minD = minDistance(uv);
	//minD = 1.0 - minD;    // Inverso
	//vec3 color = vec3(minD*.93, minD*.23, minD*.13);
	vec3 color = vec3(minD*uv.x, minD*uv.y, minD*uv.x);
	gl_FragColor = vec4(color ,1.0);
}