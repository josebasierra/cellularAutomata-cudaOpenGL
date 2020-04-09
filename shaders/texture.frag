#version 330

uniform vec4 color;
uniform sampler2D tex;

uniform vec2 useColor;

in vec2 texCoordFrag;
out vec4 outColor;

void main()
{
	// Discard fragment if texture sample has alpha < 0.5
	// otherwise compose the texture sample with the fragment's interpolated color
	vec4 texColor = texture(tex, texCoordFrag);
	if(texColor.a < 0.5f)
		discard;
	outColor = texColor;

	if(useColor.x == 1.f) outColor = color*texColor; 
}

