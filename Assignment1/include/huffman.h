
#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <stdio.h>
#include <stdint.h>
#include <queue>
#include <map>

const uint64_t HashCapacity = 256;

#define HUFFMAN_SUCCESS 0
#define HUFFMAN_FAILURE -1
#define HUFFMAN_EFAIL_MEMORY -2
#define HUFFMAN_INTERNAL_NODE_ID	300
#define HUFFMAN_PSEOF				126
#define HUFFMAN_MAX_STRLEN			27
#define HEADER_START				'{'
#define HEADER_END					'}'
#define INVALID_DATA                255
#define HEADER_INVALID_BYTE          1

typedef int8_t HUFFMAN_RESULT;
/**
 * @param char*         A char to get the ASCII val.
 * @return ASCII VAl;
 **/

uint16_t hash_function(char* str);

/**
 * @brief Define hash table item structures
 */

typedef struct huff_node_s
{
	uint16_t*      key; // Symbol ASCII
    char*         value; // corresponding ascii val
	uint64_t     count; // corresponding count
}huff_node;

/**
 * @brief Histogram (HASH Table of all symbols) 
 * 
 */

typedef struct Huffman_Hash_Table_s
{
	huff_node** items;
	uint64_t    size_table;
	uint64_t     count; 
} Huffman_Hash_Table;

typedef struct Huffman_Pqueue_node_s
{
	huff_node* p_item;
	char* p_item_binary; //binary representation
	huff_node* left;
	huff_node* right;
	uint16_t priority;
} Huffman_Pqueue_node;

typedef struct Huffman_sort_node_s
{	
	uint16_t array_id;
	uint16_t ascii_id;
	uint16_t item_count;
	struct Huffman_sort_node_s* left;
	struct Huffman_sort_node_s* right;
}Huffman_sort_node;

//Referenced from Geeks for Geeks
class HuffmanTreeNode 
{
public:
	// array id
	uint16_t array_id; // just to see the history of original element for debugging
    // Stores ascii
    uint16_t ascii_id;
 
    // Stores frequency of
    // the character
    uint16_t item_freq;
 
    // Left child of the
    // current node
    HuffmanTreeNode* left;
 
    // Right child of the
    // current node
    HuffmanTreeNode* right;
 
    // Initializing the
    // current node default constructor
    HuffmanTreeNode(uint16_t arr_id, 
                    uint16_t asc_id,
                    uint16_t freq)
    {
        array_id = arr_id;
		ascii_id = asc_id;
        item_freq = freq;
        left = NULL;
        right = NULL;
    }
};

//Referenced from Geeks for Geeks
// Custom comparator class
class Compare
{
public:
    bool operator()(HuffmanTreeNode* a,
                    HuffmanTreeNode* b)
    {
        // Defining priority on
        // the basis of frequency, having lesser frequency item to be stored first
        return a->item_freq > b->item_freq;
    }
};


HUFFMAN_RESULT create_huff_node(uint16_t key, char* value, huff_node** node);
HUFFMAN_RESULT create_huffman_hashtable(uint64_t size, Huffman_Hash_Table** table);
void           free_huffman_node(huff_node* item);
void           free_huffman_hash_table(Huffman_Hash_Table* table);
HUFFMAN_RESULT huff_node_insert(Huffman_Hash_Table* table, uint16_t* key, char* value);
HUFFMAN_RESULT huffman_node_increment_count(Huffman_Hash_Table* table, uint16_t key);
HUFFMAN_RESULT huffman_node_search(Huffman_Hash_Table* table, char* value, uint16_t* key);
void           print_huffman_item(Huffman_Hash_Table* table, char* val);
void           print_huffman_table(Huffman_Hash_Table* table);
HUFFMAN_RESULT huffman_create_sorting_data(Huffman_Hash_Table* table, Huffman_sort_node** sorting_data);
HUFFMAN_RESULT huffman_hash_table_sort(Huffman_sort_node* table, uint16_t table_size);
HUFFMAN_RESULT huffman_radix_sort(Huffman_sort_node* a, uint16_t n);
HUFFMAN_RESULT huffman_count_sort(Huffman_sort_node*a, uint16_t n, uint16_t pos);
uint16_t       get_max_item_count(Huffman_sort_node*a, uint16_t n);
void           print_huffman_codes(Huffman_sort_node* root, int code[], int cur_pos);
void           free_input_sort_node(Huffman_sort_node*input_sort_data, uint16_t length);
HUFFMAN_RESULT huffman_build_tree_pq(HuffmanTreeNode** root, 
    std::priority_queue<HuffmanTreeNode*,
    std::vector<HuffmanTreeNode*>,
    Compare> pq);
void           print_huffman_codes_pq(HuffmanTreeNode* root, char code[], int cur_pos);
void           store_huffman_code_map(HuffmanTreeNode* root, std::map<uint16_t, std::string>&huff_code_map, std::string str);
void           print_huffman_code_map(std::map<uint16_t, std::string>huff_code_map);
void           print_huffman_decode_map(std::map< std::string, uint16_t>huff_decode_map);
/**
 * Header start Ascii Val-> 16 (Syn)
 *
 * 
 * Symbol format (Ascii, Frequency, Binary)
 * Header End Ascii val->   15(Nak)
 * Eg:Header+Payload
 * syn(Ascii,Frequency,Binary), ()...Nak
*/
void           create_huffman_code_header(std::map<uint16_t, std::string>huff_code_map, std::string &str);

void           encode_data(std::map<uint16_t,std::string>huff_code_map, std::string &str,
const unsigned char* bufin, uint32_t bufinlen);

void encode_data_byte_form(std::string& header, std::vector<unsigned char>& vector_enc_data);
/**
 * @param bufin       Array of characters to encode
 * @param bufinlen    Number of characters in the array
 * @param pbufout     Pointer to unallocated output array
 * @param pbufoutlen  Pointer to variable where to store output size
 *
 * @return error code (0 is no error)
 **/
int huffman_encode(const unsigned char *bufin,
		   uint32_t bufinlen,
		   unsigned char **pbufout,
		   uint32_t *pbufoutlen);


/**
 * @param bufin       Array of characters to decode
 * @param bufinlen    Number of characters in the array
 * @param pbufout     Pointer to unallocated output array
 * @param pbufoutlen  Pointer to variable where to store output size
 *
 * @return error code (0 is no error)
 **/
int huffman_decode(const unsigned char *bufin,
  		   uint32_t bufinlen,
		   unsigned char **bufout,
		   uint32_t *pbufoutlen);


HUFFMAN_RESULT huffman_decode_create_map(const unsigned char* pstr, 
    uint16_t inlength, std::map<uint16_t, 
    std::string>&huff_code_map,std::map<std::string,
    uint16_t>&huff_decode_map, 
    uint16_t* idx);

HUFFMAN_RESULT huffman_decode_create_tree(HuffmanTreeNode** root,
std::map<uint16_t, std::string>huff_code_map);

HUFFMAN_RESULT huffman_decode_input(HuffmanTreeNode* root,
    const unsigned char *bufin,
    uint32_t bufinlen,
    //uint16_t header_lenght,
    unsigned char **bufout,
    uint32_t *pbufoutlen);

HUFFMAN_RESULT huffman_decode_convert_byte_bitstream(const unsigned char* pinput_str, uint64_t length, 
uint64_t start_length,std::string& bitstream, 
uint8_t invalid_num);

#endif