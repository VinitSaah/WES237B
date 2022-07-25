#include "huffman.h"
#include <stdlib.h>
#include <iostream>

uint32_t set_bit(uint8_t in_data, uint8_t bit_pos);
bool is_bit_set(uint8_t in_data, uint8_t bit_pos);

uint32_t set_bit(uint8_t in_data, uint8_t bit_pos)
{
	in_data |= (1 << bit_pos);
	return in_data;
}

bool is_bit_set(uint8_t in_data, uint8_t bit_pos)
{
	bool retval = false;
	in_data &= (1 << bit_pos);
	if(in_data)
	{
		retval = true;
	}
	return retval;
}

void convert_byte_to_bin(uint8_t data, std::string& str)
{
	for(uint8_t i = 0; i < 7; i++)
	{
	    if(is_bit_set(data, i))
		{
			str.push_back('1');
		}
		else
		{
			str.push_back('0');
		}
	}
}

uint16_t hash_function(char* str)
{
	uint16_t ascii_val = 0;
	ascii_val = *str;
	//std::cout << "hash_function Ascii Value= " << ascii_val << " Char = " << *str << " Mod value = " << ascii_val % HashCapacity<< std::endl;
	return ascii_val % HashCapacity;
}

HUFFMAN_RESULT create_huff_node(uint16_t key, char* value, huff_node** item)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;

	huff_node* node = (huff_node*) malloc(sizeof(huff_node));
	if (NULL == node)
	{
		std::cout << "Node creation failed\n";
		retval = HUFFMAN_EFAIL_MEMORY;
	}
	else
	{
		node->key = (uint16_t*) malloc(sizeof(uint16_t));
		if(NULL == node->key)
		{
			std::cout << "Memory allocation error\n"; 
			retval = HUFFMAN_EFAIL_MEMORY;
		}
		else
		{
			*(node->key) = key;
		}

		node->value = (char*) malloc(sizeof(char));
		if(NULL == node->value)
		{
			std::cout << "Memory allocation error\n"; 
			retval = HUFFMAN_EFAIL_MEMORY;
		}
		else
		{
			*(node->value) = *value;
			node->count = 1; //default count;
			*item = node;
		}
	}
	std::cout << "create_huff_node Node Key= " << *(node->key) << " Node Value = " << *(node->value) << std::endl;
	std::cout << "create_huff_node item key = " << *(*item)->key << " Item value = " << *(*item)->value << std::endl;
	return retval;
}

HUFFMAN_RESULT huff_node_increment_count(huff_node* node)
{
	HUFFMAN_RESULT retval = HUFFMAN_FAILURE;
	if (NULL != node)
	{
		node->count++;
		retval = HUFFMAN_SUCCESS;
	}
	return retval;
}

HUFFMAN_RESULT create_huffman_hashtable(uint64_t size, Huffman_Hash_Table** table)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS; 

	Huffman_Hash_Table* hash_table = (Huffman_Hash_Table*) malloc(sizeof(Huffman_Hash_Table));
	if (NULL == hash_table)
	{
		std::cout << "Memory error\n";
		retval = HUFFMAN_EFAIL_MEMORY;
	}
	else
	{
	    hash_table->size_table = size;
	    hash_table->count = 0;
		hash_table->items = (huff_node**) malloc(hash_table->size_table* sizeof(huff_node));//[TODO: Correct the size of it]
		if(NULL == hash_table->items)
		{
			std::cout << "Memory error \n";
			retval = HUFFMAN_EFAIL_MEMORY;
		}
		else
		{
			memset(hash_table->items, 0, hash_table->size_table* sizeof(huff_node));//[TODO: Correct the size of it]
			*table = hash_table;
			std::cout << "create_huffman_hashtable Table Address = " << *table << " Table Item Address = " << hash_table->items << std::endl;
		}
	}
	return retval;
}

void free_huffman_node(huff_node* item)
{
	if (NULL != item)
	{
		if(NULL!= item->key)
		{
			free(item->key);
		}
		if(NULL!= item->value)
		{
			free(item->value);
		}
		item->count = 0;
		free(item);
	}
}

void free_huffman_hash_table(Huffman_Hash_Table* table)
{
    huff_node* temp_node = NULL;
	if(NULL != table && NULL != table->items)
	{
		for (int i = 0; i<table->size_table; i++)
		{
			if (table->items[i]!= NULL)
			{
				temp_node = table->items[i];
				free_huffman_node(temp_node);
			}
		}
		free(table->items);
		free(table);
	}
}

HUFFMAN_RESULT huff_node_insert(Huffman_Hash_Table* table, uint16_t* key, char* value)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;
	if(table == NULL)
	{
		retval = HUFFMAN_FAILURE;
		std:: cout << "Null pointer  huff man table\n";
	}
	else
	{
		/** Check if item already exist, increment count if it exist
		 * else create a new node
		 */
		if(table->items[*key] == NULL)
		{
			std:: cout << "Inserting new element\n";
			retval = create_huff_node(*key, value, &(table->items[*key]));
			table->count++;
		}
		else
		{
			//std:: cout << "Incrementing count\n";
			retval = huffman_node_increment_count(table, *key);
		}
		
	}
	return retval;
}

HUFFMAN_RESULT huffman_node_increment_count(Huffman_Hash_Table* table, uint16_t key)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;
	if(table == NULL)
	{
		retval = HUFFMAN_FAILURE;
		std:: cout << "Null pointer  huff man table\n";
	}
	else
	{
        table->items[key]->count++;
		//std::cout<< "Item count = " <<table->items[key]->count << std::endl;
	}

	return retval;
}

HUFFMAN_RESULT huffman_node_search(Huffman_Hash_Table* table, char* value, uint16_t* key)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;
	uint16_t temp_key = 0;
	if(table == NULL || value == NULL)
	{
		retval = HUFFMAN_FAILURE;
		std:: cout << "Null pointer  huffman table\n";
	}
	else
	{
		temp_key = hash_function(value);
		if(key !=NULL)
		{
			*key = temp_key;
			std::cout << "Key value= " << *key << std::endl; 
		}
	}
    return retval;
}

void print_huffman_item(Huffman_Hash_Table* table, char* val)
{
	uint16_t key = 0;
	huffman_node_search(table, val, &key);
	std::cout<< "print_huffman_item " << "Key = " << key <<std::endl;
	//if(NULL != table->items[key] && *(table->items[key]->value) == *val)
	if(NULL != table->items[key])// && *(table->items[key]->value) == *val)
	{
		std::cout << "Key = " << key << " Value =" << *val << " count = " << table->items[key]->count << std::endl;
	}
	else
	{
		if(table->items[key] == NULL)
		{
			std:: cout << "Table -> Item[key] is null\n";
		}
		else
		{
			std::cout << "Table-> Item[key]->value=\n" << table->items[key]->value;	
		}
		std:: cout<< "Char not found\n";
	}

}

void print_huffman_table(Huffman_Hash_Table* table)
{
	std::cout << "____Huffman Hashtable____\n";
	std::cout << "S.no " <<"	"<<"Key" << "	" << "Value" << "	"<<"Count\n";
	std::cout << "______|________|______|____\n";
	for(int i =0; i < HashCapacity; i++)
	{
		if(NULL != table->items[i])
		{
			std::cout << i<<"	"<< *(table->items[i]->key)<<"	"<<*(table->items[i]->value) << "	"<< table->items[i]->count << std::endl;
		}
	} 
}

HUFFMAN_RESULT huffman_create_sorting_data(Huffman_Hash_Table* table, Huffman_sort_node** sorting_data)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;
	Huffman_sort_node* temp_node = NULL; 
	if(NULL == table)
	{
		std::cout << "Hash table is null\n";
		retval = HUFFMAN_FAILURE;
	}
	else
	{
	    temp_node = (Huffman_sort_node*) malloc(table->count * sizeof(Huffman_sort_node));
		if (NULL == temp_node)
		{
			std::cout << "Huffman Sorting input data Memory error\n";
			retval = HUFFMAN_EFAIL_MEMORY;
		}
		else
		{
			int j = 0;
			for (int i = 0; i < HashCapacity; i++)
			{
			    if(NULL != table->items[i])
			    {
				    temp_node[j].array_id = j;
					temp_node[j].ascii_id = i;
					temp_node[j].item_count = table->items[i]->count;
					temp_node[j].left = NULL;
					temp_node[j].right = NULL;
					++j;
				}
			}
			*sorting_data = temp_node;
		}
	}
	return retval;
}
HUFFMAN_RESULT huffman_hash_table_sort(Huffman_sort_node* table, uint16_t table_size)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS; 
	if (NULL == table || table_size <= 0 || table_size > 255) 
	{
		std::cout << "Memory access error or table size error";
		retval = HUFFMAN_FAILURE;
	}
	
	retval = huffman_radix_sort(table, table_size);

	for(int i = 0; i< table_size; i++)
	{
		std::cout << "Sorted entries " << table[i].array_id << "= " << table[i].item_count <<" Ascii = " << table[i].ascii_id<< std::endl;
	}

	// copy the priority queue
	return retval;
}

HUFFMAN_RESULT huffman_radix_sort(Huffman_sort_node* a, uint16_t n)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;
	
	int16_t max_count = get_max_item_count(a, n);
	std::cout << "Max Elemental Item Count = " << max_count << std::endl;

	for(uint16_t pos = 1; max_count/pos > 0; pos *= 10)
	{
		retval = huffman_count_sort(a, n, pos);
	}

	return retval;
}

HUFFMAN_RESULT huffman_count_sort(Huffman_sort_node*a, uint16_t n, uint16_t pos)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;

	uint16_t count[10] = {0}; 
	Huffman_sort_node* b = NULL; // temporary array

	b = (Huffman_sort_node*) malloc(n*sizeof(Huffman_sort_node));
	memset(&b[0], 0, n*sizeof(Huffman_sort_node));

	//Find count of digits and increment respective index content by 1;
	for(int i = 0; i< n; i++)
	{
		++count[(a[i].item_count/pos) % 10]; // find the positional digit at content of a[i].item_count
	}
	
	//Find Prefix sum
	for (int i = 1; i < 10; i++)
	{
		count[i] = count[i] + count[i-1];
	}

	//Map the item in 'a' starting from right to maintain stability.(Element coming first would be reported first)
	for(int i = n-1; i >=0; i--)
	{	
		int idx = --count[(a[i].item_count/pos)%10];
		b[idx].item_count = a[i].item_count;
		b[idx].array_id = idx;
		b[idx].ascii_id = a[i].ascii_id;
	}

	for(int i = 0; i< n; i++)
	{
		a[i].array_id = b[i].array_id;
		a[i].ascii_id = b[i].ascii_id;
		a[i].item_count = b[i].item_count;
	}
	free(b);
	return retval;
}

uint16_t get_max_item_count(Huffman_sort_node*a, uint16_t n)
{
	uint16_t max_item = a[0].item_count;
	if (NULL != a)
	{
        for (int i = 1; i < n; i++)
            if (a[i].item_count > max_item)
		    {
                max_item = a[i].item_count;
		    }
	}
	return max_item;
}

/**
 * TODO Complete this function
 **/
int huffman_encode(const unsigned char *bufin,
						  unsigned int bufinlen,
						  unsigned char **pbufout,
						  unsigned int *pbufoutlen)
{
	char val = '\0';
	uint16_t key                       = 0;
	Huffman_Hash_Table* hash_table     = NULL;
	Huffman_sort_node* input_sort_data = NULL;
	//Huffman_sort_node* root            = NULL;
	HuffmanTreeNode* root              = NULL;
	unsigned char* p_encoded_data                 = NULL;
	//Huffman_sort_node* tmp_root        = NULL;
	//uint16_t           height          = 0;
	//int			   bin_sym[10];
	//uint16_t cur_pos = 0;
	
	create_huffman_hashtable(HashCapacity, &hash_table);

	for(int64_t i = 0; i< bufinlen;i++)
	{
		val = bufin[i];
	    key = hash_function(&val);
	    huff_node_insert(hash_table, &key, &val);
	}
	print_huffman_table(hash_table);
	huffman_create_sorting_data(hash_table, &input_sort_data);
	for(int i = 0; i< hash_table->count; i++)
	{
		std::cout << "Input entries " << i << "=" << input_sort_data[i].item_count << std::endl;
	}

	huffman_hash_table_sort(input_sort_data, hash_table->count);

	std::priority_queue<HuffmanTreeNode*,
                   std::vector<HuffmanTreeNode*>,
                   Compare> pq;
//#if 0
	//create a pseudo EOF so that we could mark end of decoding
	HuffmanTreeNode* newNode
            = new HuffmanTreeNode(HUFFMAN_PSEOF, HUFFMAN_PSEOF,1); //character with ASCII value 254, part of extended ASCII table.
        pq.push(newNode);
//#endif
	for(int i = 0; i < hash_table->count; i++)
	{
		HuffmanTreeNode* newNode
            = new HuffmanTreeNode(input_sort_data[i].array_id, input_sort_data[i].ascii_id,
			input_sort_data[i].item_count);
        pq.push(newNode);
	}

	// free input sort node; no need of it after getting the Priority Queue.
	free_input_sort_node(input_sort_data, hash_table->count);

	//Create Huffman code tree
	//using priority queue method
	huffman_build_tree_pq(&root, pq);
	std::cout << "huffman_tree root 1 " << root <<"\n"; 

    //int arr[HUFFMAN_MAX_STRLEN] = {0};
	char arr[HUFFMAN_MAX_STRLEN];
	memset(arr, 0, HUFFMAN_MAX_STRLEN*sizeof(char));
	int top = 0;
    //print_huffman_codes_pq(root, arr, top);
	print_huffman_codes_pq(root, arr, top);
	//std::cout << "huffman_tree root 2 " << root <<"\n"; 

	// Store Encoding Map
	std::map<uint16_t, std::string> code_map;
	store_huffman_code_map(root, code_map, "");
	//std::cout << "huffman_tree root 3 " << root <<"\n"; 
	print_huffman_code_map(code_map);
	
	std::string header;
	create_huffman_code_header(code_map,header);
	std::cout << "Header Size = " << header.length() << std::endl;

	//Create Encoded data
	encode_data(code_map, header, bufin,bufinlen);
	std::cout << "Header\n";
	std::cout << header << std::endl;

	//now we have header and bit streams in string 
	//store byte information on vector

	std::vector<unsigned char> vector_enc_data;
	encode_data_byte_form(header, vector_enc_data);
	std::cout << "****Vector compress data creation****\n";
	for(int i = 0; i <vector_enc_data.size(); i ++)
	{
		std::cout << vector_enc_data[i];
	}
	std::cout << std::endl;
	std::cout << "Compressed size = " << vector_enc_data.size() << std::endl;
	uint16_t coded_size = vector_enc_data.size();
	p_encoded_data = (unsigned char*)malloc(sizeof(unsigned char)*(coded_size));

	//std::cout << std::endl;
	for(int i = 0; i < coded_size ; i++)
	{
		p_encoded_data[i] = vector_enc_data[i];
		std::cout<< p_encoded_data[i];
	}
	std::cout << "\n";
	std::cout << "Footer\n";
	

	*pbufout = p_encoded_data;
	*pbufoutlen = coded_size;

	free_huffman_hash_table(hash_table);

	return 0;
}


uint16_t get_min(uint16_t i, uint16_t j)
{	
	if(i <= j)
	{
	    return i;
	}
	else
	{
		return j;
	}

}

HUFFMAN_RESULT huffman_build_tree_pq(HuffmanTreeNode** root, 
    std::priority_queue<HuffmanTreeNode*,
    std::vector<HuffmanTreeNode*>,
    Compare> pq)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;
	std::cout << "pq size = " << pq.size() << "\n";
    while (pq.size() != 1) 
	{
        HuffmanTreeNode* left = pq.top();
        // Remove node
        pq.pop();
        HuffmanTreeNode* right = pq.top();
        pq.pop();
		uint16_t count = left->item_freq + right->item_freq;
		uint16_t id = 0;//fill 0 not used anywhere
		uint16_t asc_id = HUFFMAN_INTERNAL_NODE_ID;
        HuffmanTreeNode* node = new HuffmanTreeNode(id,asc_id,
                                  count);
        node->left = left;
        node->right = right;
 
        // Push back node
        pq.push(node);
    }
 
    *root = pq.top();
	//std::cout<< "pq.top address" << pq.top() << "\n"; 
	return retval;
}
//void print_huffman_codes_pq(HuffmanTreeNode* root,int arr[], int top)

void print_huffman_codes_pq(HuffmanTreeNode* root,char arr[], int top)
{
	HuffmanTreeNode* node = root; 
    if(NULL == root)
	{
		std::cout << "Returning, Null root encountered\n";
       return;
    }
	// Assign 0 to the left node
    // and recur
    if (node->left)
	{
        arr[top] = '0';
        print_huffman_codes_pq(node->left,
                   arr, top + 1);
    }
 
    // Assign 1 to the right
    // node and recur
    if (node->right) 
	{
        arr[top] = '1';
        print_huffman_codes_pq(node->right, arr, top + 1);
    }
 
    // If this is a leaf node,
    // then we print root->data
 
    // We also print the code
    // for this character from arr
    if (NULL == node->left && NULL == node->right) 
	{
        std::cout << node->array_id << "--" <<node->ascii_id << "-> " << (unsigned char)(node->ascii_id) << "-- ";
        for (int i = 0; i < top; i++) 
		{
            std::cout << arr[i];
        }
        std::cout << std:: endl;
    }
}

void store_huffman_code_map(HuffmanTreeNode* root, std::map<uint16_t, std::string>&huff_code_map, std::string str)
{
	if (root==NULL)
	{
		return;
	} 
	if (root->ascii_id != HUFFMAN_INTERNAL_NODE_ID && root->left == NULL && root->right == NULL)
	{
		huff_code_map[root->ascii_id]=str;
	}
	store_huffman_code_map(root->left, huff_code_map, str + "0"); 
	store_huffman_code_map(root->right, huff_code_map, str + "1"); 
}

void print_huffman_code_map(std::map<uint16_t, std::string>huff_code_map)
{
	std:: cout << "Ascii value " << " --- " << " Symbol " << " -- " << " Code " << std::endl;
	std::map<uint16_t, std::string>::iterator itr;
	for(itr = huff_code_map.begin(); itr != huff_code_map.end(); ++itr)
	{
		std:: cout << itr->first << "		" << (char)itr->first << "		" <<itr->second << std::endl;
	}
}

void print_huffman_decode_map(std::map<std::string, uint16_t>huff_decode_map)
{
	std::cout << "Generated Decode Table\n";
	std:: cout << "Symbol " << " ------ " << " Ascii symbol" << std::endl;
	std::map<std::string, uint16_t>::iterator itr;
	for(itr = huff_decode_map.begin(); itr != huff_decode_map.end(); ++itr)
	{
		std:: cout << itr->first << "		" <<(char)itr->second << std::endl;
	}
}

void create_huffman_code_header(std::map<uint16_t, std::string>huff_code_map, std::string &str)
{
	/** Header format
	 * Header start Ascii Val-> {
	 *
	 * 
	 * Symbol format (Ascii, *Frequency*, Binary) // frequency is optional
	 * Header End Ascii val->   }
	 * Header+Payload
	 * syn(Ascii,Frequency,Binary), ()...Nak
	 */
	char h_st = HEADER_START;
	char h_end = HEADER_END;
	str.push_back(h_st);
	//std::cout << str;
	std::map<uint16_t, std::string>::iterator itr;
	const char* p_symbol = NULL;
	for(itr = huff_code_map.begin(); itr != huff_code_map.end(); ++itr)
	{
		str.push_back('(');
		str.push_back((char)itr->first);
		str.push_back(',');
		p_symbol = itr->second.c_str();
		for(int i = 0; i < itr->second.length();i++)
		{
			str.push_back(*(p_symbol+i));
		}
		str.push_back(')');
		//std::cout << str;
	}
	str.push_back(h_end);
}

void encode_data(std::map<uint16_t,std::string>huff_code_map, std::string &str, 
const unsigned char* bufin, uint32_t bufinlen)
{
	const char* p_symbol = NULL;
	//read byte by byte, get ascii value, get equivalent binary, convert, store
	std::map<uint16_t, std::string>::iterator itr;
	uint16_t marker = HUFFMAN_PSEOF;
	for(int i = 0; i < bufinlen;i++)
	{
        itr = huff_code_map.find((uint16_t)bufin[i]);
		p_symbol = itr->second.c_str();
		for(int i = 0; i < itr->second.length();i++)
		{
			str.push_back(*(p_symbol+i));
		}
	}

//#if 0
	//append marker so that it could be used while decoding
	itr = huff_code_map.find(marker);
	p_symbol = itr->second.c_str();
	for(int i = 0; i < itr->second.length();i++)
	{
		str.push_back(*(p_symbol+i));
	}
//#endif
	//std::cout << str << std::endl;
	return;
}

void encode_data_byte_form(std::string header, std::vector<unsigned char>& vector_enc_data)
{
	uint32_t idx_cur = 0;
	//store header char by char
	for(uint32_t i = 0; i < header.length(); i++)
	{
		vector_enc_data.push_back(header[i]);
		if(header[i] == '}')
		{
			//filled header break
			idx_cur = i;
			break;
		}
	}
	idx_cur++; //go to bitstreams
	uint8_t temp_data = 0;
	//Now read char by char and create a byte and add to vector_enc_data
	uint64_t read_rem = header.length()-idx_cur; //length-header
	while(read_rem)
	{
		temp_data = 0;
		int loop_count = 8; //for a byte of data
		if(read_rem % 8)
		{
			loop_count = (read_rem%8);
			std::cout << "Read remaining loop count updated to " << loop_count << std::endl;
		}
	    for(int j = 0; j < loop_count; j++)
		{
			if(header[idx_cur+j] == '1')
			{
				temp_data = set_bit(temp_data, j);
			}
		}
		idx_cur += loop_count;
		read_rem -= loop_count;  
		//store byte to vector
		vector_enc_data.push_back(temp_data);
	}
}

bool is_leaf(Huffman_sort_node* node)
{
	if (NULL!= node)
	{
		if(NULL== node->left && NULL == node->right)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
}

void print_symbols(int code[], int cur_pos)
{
    int i;
    for (i = 0; i < cur_pos; ++i)
	{
        std::cout << code[i];
	}
    std::cout << std::endl;
}

void print_huffman_codes(Huffman_sort_node* root, int code[], int cur_pos)
{
	  //If root is Null then return.
       if(NULL == root)
	   {
		   std::cout << "Returning, Null root encountered\n";
           return;
       }

	   if (root->left) 
		{
            code[cur_pos] = 0;
            print_huffman_codes(root->left, code, cur_pos + 1);
        }
        
		if (root->right) 
		{
            code[cur_pos] = 1;
            print_huffman_codes(root->right, code, cur_pos + 1);
        }

       //If the node's data is not HUFFMAN_INTERNAL_NODE_ID that means it's not an internal node and print the string.
       if(is_leaf(root))
	   {
          std::cout << "root-data = " << (char)root->ascii_id << std::endl;
		  print_symbols(code, cur_pos);
       }
}

void free_input_sort_node(Huffman_sort_node*input_sort_data, uint16_t length)
{
	if(input_sort_data)
	{
		for(int i = 0; i < length; i++)
		{
			input_sort_data[i].left = NULL;
			input_sort_data[i].right = NULL;
		}
		free(input_sort_data);
	}
}
/**
 * TODO Complete this function
 **/
int huffman_decode(const unsigned char *bufin,
						  unsigned int bufinlen,
						  unsigned char **pbufout,
						  unsigned int *pbufoutlen)
{
	std::cout << "*********  Decode   *******\n";
	HuffmanTreeNode* root              = NULL;
	//create ascii value and symbol map.
	std::map<uint16_t, std::string>code_map;
	std::map<std::string,uint16_t>decode_map;
	uint16_t cur_idx = 0;

	huffman_decode_create_map(bufin, bufinlen, code_map, decode_map,&cur_idx);
	print_huffman_code_map(code_map);
	print_huffman_decode_map(decode_map);
	
	//create the huffman coding tree
	huffman_decode_create_tree(&root, code_map);
	std::cout<<"Decoder root address = " << root << std::endl;

	char arr[HUFFMAN_MAX_STRLEN];
	memset(arr, 0, HUFFMAN_MAX_STRLEN*sizeof(char));
	int top = 0;
	print_huffman_codes_pq(root, arr, top);
	//std::cout<<"Decoder root address after print = " << root << std::endl;
	
	//decode the input.
	huffman_decode_input(root,&bufin[0],bufinlen, pbufout, pbufoutlen);
	return 0;
}

HUFFMAN_RESULT huffman_decode_create_map(const unsigned char* pstr,
 uint16_t inlength, std::map<uint16_t, 
 std::string>&huff_code_map,
 std::map<std::string, uint16_t>&huff_decode_map, 
 uint16_t* cur_idx)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;

	/** Parse char by char and create encode map */
	for(int i = 0; i < inlength; i++)
	{
		if(pstr[i] == '{') //start of header
		{
			std::cout << "Header Start \n";
			continue;
		}
		if(pstr[i] == '(')
		{
			std :: string bin_sym = ""; 
			for(int j = i+3;;j++ )
			{
				if(pstr[j] == ')')
				{
					break;
				}
				
			    bin_sym.push_back((char)pstr[j]);
			}
			huff_code_map[pstr[i+1]] = bin_sym;
			huff_decode_map[bin_sym] = pstr[i+1];
		}
		if(pstr[i] == '}')
		{
			std::cout << "Header creation successful\n";
			*cur_idx = i;
			break;
		}
	}
	return retval;
}

HUFFMAN_RESULT huffman_decode_create_tree(HuffmanTreeNode** root, 
    std::map<uint16_t, std::string>huff_code_map)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;
	HuffmanTreeNode* tree_root = NULL;
	std::map<uint16_t, std::string>::iterator itr;

	//create 1st root
	uint16_t id = 0;
	uint16_t asc_id = HUFFMAN_INTERNAL_NODE_ID;
	uint16_t count = 0;
	HuffmanTreeNode* node = new HuffmanTreeNode(id,asc_id,count);
	tree_root = node;
	std::cout << "Creating Decoder Tree , node address = " << node  << std::endl;
	for(itr = huff_code_map.begin(); itr != huff_code_map.end(); ++itr)
	{
		node = tree_root;
		//std::cout << "Going back to Root Node address = " << node <<std::endl;
		//std::cout << "Root Ascii " << node->ascii_id << std::endl;
		for(int i = 0; i < itr->second.length(); ++i)
		{
			if(itr->second[i] ==  '0')
			{
				if(node->left == NULL)
				{
					node->left = new HuffmanTreeNode(id,asc_id,count);
					//std::cout << "Creating new intermediate node  = " << node->left << "\n";
				}
				else
				{
					//std::cout << "Intermediate node exist = " << node->left << "\n";
				}
				node = node->left;
				//std::cout << "Intermediate address " << node << "Intermediate Ascii " << node->ascii_id << std::endl;
			}
			else if(itr->second[i] ==  '1')
			{
				if(node->right == NULL)
				{
					node->right = new HuffmanTreeNode(id,asc_id,count);
					//std::cout << "Creating new intermediate node  = " << node->right << "\n";
				}
				else
				{
					//std::cout << "Intermediate node exist = " << node->right << "\n";
				}
				node = node->right;
				//std::cout << "Intermediate address " << node << "Intermediate Ascii " << node->ascii_id << std::endl;
			}
			else
			{
				std::cout << "Incorrect symbol" << (int)itr->second[i]<<"\n";
				retval = HUFFMAN_FAILURE;
			}
		}
		node->ascii_id = itr->first;
		//std::cout << "Leaf address " << node << " Leaf Ascii " << node->ascii_id << std::endl;
	}
	*root = tree_root;
	return retval;
}

HUFFMAN_RESULT huffman_decode_input(HuffmanTreeNode* root,
    	const unsigned char *bufin,
    	uint32_t bufinlen,
		//uint32_t header_length,
    	unsigned char **bufout,
    	uint32_t *pbufoutlen)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;
	HuffmanTreeNode* cur_node = NULL;
    HuffmanTreeNode* tree_root = NULL;
	unsigned char cur_bit = '\0';
	uint16_t cur_pos = 0;
	std::cout << bufinlen << std::endl;
	std:: string decoded_data = "";

	//Iterate over input to cross header
	for(int i = 0; i < bufinlen; i++)
	{
		if(bufin[i] == '}')
		{
			std:: cout << "Header Parsed " << bufin[i]<< std::endl;
			cur_pos = i;
			break;
		}

	}
	cur_pos++; //start of symbols;
	std::cout << "Bit stream to be decoded \n";
	for(int i = cur_pos; i < bufinlen ; i++)
	{
		std :: cout << bufin[i];
	} 
	//std::cout << bufin[cur_pos] << std::endl;
	std::cout << std::endl;

	tree_root = root;
	std::cout << "****Decoding start*****\n";
	std::cout << "Root address = " << root << std::endl;
	cur_node = tree_root;
	for(uint16_t idx = cur_pos;idx < bufinlen; idx++)
	{	
		std::string str = "";
		cur_bit = bufin[idx];	
		//std :: cout << "Current bit = " << cur_bit << std::endl;
		if (cur_bit == '0')
		{
			//std :: cout << "Left--"; 
        	cur_node = cur_node->left;
			str.push_back('0');
		}
      	else if (cur_bit == '1')
		{
			//std :: cout << "Right--"; 
			cur_node = cur_node->right;
			str.push_back('1');
		}
      	else
		{
        	std::cerr << "Undefined bit: " << cur_bit << " -- check input\n";
				//break;
		}
		if(NULL == (cur_node->left) && NULL == (cur_node->right))
		{
			//std::cout << std::endl;
			//std::cout << "Reached a leaf" <<std::endl;
			//std::cout << "cur_node->ascii_id" << std::endl;
			//std::cout <<"Getting back to root " << root << std::endl;
			if(HUFFMAN_PSEOF != cur_node->ascii_id) // reached EOF
			{
			    decoded_data.push_back(char(cur_node->ascii_id));
			}
			//std::cout << cur_node->ascii_id;
			cur_node = tree_root;
		}
		else
		{
			//std::cout << cur_node << " child " << cur_node->left  << "--" <<  cur_node->right<< std::endl;
		}
	}
	std::cout << decoded_data << std::endl;
	unsigned char* pBuf = NULL;
	pBuf = (unsigned char*) malloc(sizeof(unsigned char)*decoded_data.length());
	for(int i = 0; i < decoded_data.length(); i++)
	{
		pBuf[i] = decoded_data[i];
	}
	*bufout = pBuf;
	
	*pbufoutlen = decoded_data.length();

	return retval;
}