
#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <stdio.h>
#include <stdint.h>

const uint64_t HashCapacity = 256;

#define HUFFMAN_SUCCESS 0
#define HUFFMAN_FAILURE -1
#define HUFFMAN_EFAIL_MEMORY -2

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
}Huffman_sort_node;

HUFFMAN_RESULT create_huff_node(uint16_t key, char* value, huff_node** node);
HUFFMAN_RESULT create_huffman_hashtable(uint64_t size, Huffman_Hash_Table** table);
void free_huffman_node(huff_node* item);
void free_huffman_hash_table(Huffman_Hash_Table* table);
HUFFMAN_RESULT huff_node_insert(Huffman_Hash_Table* table, uint16_t* key, char* value);
HUFFMAN_RESULT huffman_node_increment_count(Huffman_Hash_Table* table, uint16_t key);
HUFFMAN_RESULT huffman_node_search(Huffman_Hash_Table* table, char* value, uint16_t* key);
void print_huffman_item(Huffman_Hash_Table* table, char* val);
void print_huffman_table(Huffman_Hash_Table* table);
HUFFMAN_RESULT huffman_create_sorting_data(Huffman_Hash_Table* table, Huffman_sort_node** sorting_data);
#if 0
HUFFMAN_RESULT huffman_hash_table_sort(Huffman_Hash_Table* table, Huffman_Pqueue_node** Pqueue);
HUFFMAN_RESULT huffman_radix_sort(uint16_t* a, uint16_t n);
HUFFMAN_RESULT huffman_count_sort(uint16_t*a, uint16_t n, uint16_t pos);
uint16_t get_max_item_count(uint16_t*a, uint16_t n);
#endif
HUFFMAN_RESULT huffman_hash_table_sort(Huffman_sort_node* table, uint16_t table_size);
HUFFMAN_RESULT huffman_radix_sort(Huffman_sort_node* a, uint16_t n);
HUFFMAN_RESULT huffman_count_sort(Huffman_sort_node*a, uint16_t n, uint16_t pos);
uint16_t get_max_item_count(Huffman_sort_node*a, uint16_t n);
/** Basic data Structures 
 * 1) Basic Node
 * 2) Array vector for characters and their frequency, Histogram (Class implementation of character and their frequency)
 * 3) Sorting: Count Sort
 * 4) Use priority queue to build tree
 * 4) Traverse to find code for each char
 * 5) Examing the source file again to output the code to the destination file. 
 * 3) Sorted Array Priority Queue
 */
 


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

#endif
