figure
grs80 = almanac('earth','grs80');
ax = axesm('globe','Geoid',grs80,'Grid','on','GLineStyle','-','Gcolor','y');
set(ax,'Position',[0 0 1 1]);
view(3);
axis equal on vis3d;
set(gcf,'Renderer','opengl'); 
load topo 
geoshow(topo,topolegend,'DisplayType','texturemap');
demcmap(topo);