#pragma once

#include <SFML/Graphics.hpp>

#include "utils.h"

class renderWindow
{
  public:
    renderWindow::renderWindow() ;

    void BeginDisplay();

    void DisplayTitle(
      const int epoch,
      const double loss,
      const char* text
    );

    void DisplayImage(
      unsigned char* r,
      unsigned char* g,
      unsigned char* b,
      int x, 
      int y);

    void DisplayGrid(
      const int gridSize,
      matrix& values);

    void EndDisplay();
    
    void ProcessEvents(bool& running);

    sf::RenderWindow window;
    sf::Font font;
    sf::Text titleStr; 
    sf::Text lossStr; 
    sf::Text gradientStr; 

    sf::Texture imageTexture;
    sf::Sprite imageSprite;
     sf::Uint8* pixels;

    sf::RectangleShape rect;
};
