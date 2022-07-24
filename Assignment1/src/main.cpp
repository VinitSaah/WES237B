
#include "main.h"
#include "huffman.h"

#include <fstream>

using namespace std;


int main(int argc, const char * argv[])
{    
    cout << "WES237B Assignment 1\n";
    
	if(argc < 4)
	{
		cout << "Usage: " << argv[0] << " <input.txt> <code.txt> <output.txt>" << endl;
		return EXIT_FAILURE;
	}

	const char* in_filename   = argv[1];
	const char* code_filename = argv[2];
	const char* out_filename  = argv[3];

	// Read input
	ifstream in(in_filename);
	
	string in_contents((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());


	// Encode
	unsigned char* code = NULL;
	uint32_t code_size = 0;

	int ret;
	ret = huffman_encode((unsigned char*)in_contents.c_str(), in_contents.size(), &code, &code_size);

	if(ret != 0)
	{ cerr << "Huffman encode failed with code " << ret << endl; return EXIT_FAILURE; }


	// Save code to file
	if(code)
	{
		ofstream out_code(code_filename);
		for(uint32_t i = 0; i < code_size; i++)
		{
			out_code << code[i];
		}
	}


	// Read code from the same file
	ifstream in_code(code_filename);
	string code_contents((istreambuf_iterator<char>(in_code)), istreambuf_iterator<char>());


	// Decode
	unsigned char* decode = NULL;
	uint32_t decode_size = 0;

	ret = huffman_decode((unsigned char*)code_contents.c_str(), code_contents.size(), &decode, &decode_size);

	if(ret != 0)
	{ cerr << "Huffman decode failed with code " << ret << endl; return EXIT_FAILURE; }


	// Save output to file
	if(decode)
	{
		ofstream out(out_filename);
		out << (const char*)decode;
	}


	// Free memory
	if(code){ free(code); }
	if(decode){ free(decode); }

	// Check output
	string diff_command = "diff " + string(in_filename) + " " + string(out_filename);
	int diff_ret = system(diff_command.c_str());

	if(WEXITSTATUS(diff_ret) == 0)
	{
		cout << "SUCCESS" << endl;
		return EXIT_SUCCESS;
	}
	else
	{
		cout << "FAILURE" << endl;
		return EXIT_FAILURE;
	}

    return 0;
}
