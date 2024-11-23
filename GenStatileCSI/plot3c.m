function plot3c(x,y,z,color)
switch(color)
    case 0 
        plot3(x,y,z,'LineWidth', 20,'w-')
     case 1
        plot3(x,y,z, 'Color', [0.68, 0.85, 0.9],'LineWidth', 30)
    case 2 
        plot3(x,y,z,'b-','LineWidth', 3)
      case 3
        plot3(x,y,z,'w-','LineWidth', 3)
      case 4
        plot3(x,y,z,'m-','LineWidth', 3)
      case 5
        plot3(x,y,z,'y-','LineWidth', 2)
       case 6 
        plot3(x,y,z,'b-','LineWidth', 2)
      case 7
        plot3(x,y,z,'k-','LineWidth', 2) 
end