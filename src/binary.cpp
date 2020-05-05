#include <iostream>
#include <cstdlib>

using namespace std;

int main(int argc, char **argv)
{
    cout << argc << endl;
    if (argc != 2) {
        cout << "please enter the rule you want to use" << endl;
        return 0;
    }
    
    int rule = atoi(argv[1]);
    
    int binary[9]; //in a 2D image, the neighbourhood has 9 cells

    for(int i = 0; i < 9; i++)    
    {    
        binary[i] = rule%2;    
        rule = rule/2;  
        cout << i << " " << binary[i] << " " << rule << endl;
    }    
    
    cout<<"Binary of the given number = ";    
    for(int i = 8; i >= 0; i--)    
    {    
        cout << binary[i];    
    }    
    cout << endl;
    
    return 0;
}