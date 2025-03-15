shader_type canvas_item;
//TODO use global uniform
uniform vec2 ball_position_uv[1200];
uniform vec4 hint_color = vec4(1.0);
uniform sampler2D SCREEN_TEXTURE:hint_screen_texture,filter_linear_mipmap;

void fragment()
{
	//vec4 color = texture(SCREEN_TEXTURE,SCREEN_UV);
	//color = mix(color,hint_color,0.5);
	//COLOR = color;
	float weight = 0.0;
	for (int i = 0; i < ball_position_uv.length(); i++) {
		float dst = length(UV - ball_position_uv[i]);
		weight += 0.000015/(dst*dst);
	if(weight > 3.7){
		vec4 color = texture(SCREEN_TEXTURE,SCREEN_UV);
		vec4 water = vec4(0.15,0.58,0.78,1.0);
		color = mix(color,water,0.4);
		color = mix(vec4(0.5),color,1.5);
		COLOR =color;
		}
	else if(weight<3.7 && weight>3.4){
		COLOR = vec4(0.8,1,1,1.0);
	}
	else{
		COLOR = vec4(0,0,0,0);
	}

}
}


//void light() {
	// Called for every pixel for every light affecting the CanvasItem.
	// Uncomment to replace the default light processing function with this one.
//}
