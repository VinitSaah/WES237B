#include "huffman.h"
#include <stdlib.h>
#include <iostream>

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
		hash_table->items = (huff_node**) malloc(hash_table->size_table* sizeof(huff_node));
		if(NULL == hash_table->items)
		{
			std::cout << "Memory error \n";
			retval = HUFFMAN_EFAIL_MEMORY;
		}
		else
		{
			memset(hash_table->items, 0, hash_table->size_table* sizeof(huff_node));
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

HUFFMAN_RESULT huffman_hash_table_sort(Huffman_Hash_Table* table, Huffman_Pqueue_node** Pqueue)
{
	uint16_t entries[HashCapacity] = {0};
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS; 
	if (NULL == table) 
	{
		std::cout << "Memory access error";
		retval = HUFFMAN_FAILURE;
	}
	//fill array for radix sort
	for(int i = 0; i< HashCapacity; i++)
	{
		//if(NULL != table->items[i] && i == table->items[i]->idx)
		if(NULL != table->items[i])
		{
			entries[i] = table->items[i]->count;
		}
		else
		{
			entries[i] = 0;
		}
		//std::cout << "entries " << i << "=" << entries[i] << std::endl;
	}
    //retval = huffman_radix_sort(entries, table->count);
	retval = huffman_radix_sort(entries, table->size_table);
	for(int i = 0; i< HashCapacity; i++)
	//for(int i = 0; i< table->count; i++)
	{
		std::cout << "Sorted entries " << i << "=" << entries[i] << std::endl;
	}

	// copy the priority queue
	return retval;
}

HUFFMAN_RESULT huffman_radix_sort(uint16_t* a, uint16_t n)
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

HUFFMAN_RESULT huffman_count_sort(uint16_t*a, uint16_t n, uint16_t pos)
{
	HUFFMAN_RESULT retval = HUFFMAN_SUCCESS;

	uint16_t count[10] = {0}; 
	uint16_t* b = NULL; // temporary array

	b = (uint16_t*) malloc(n*sizeof(uint16_t));
	for(int i = 0; i < n; i++)
	{
		b[i] = 0;
	}
	//Find count of digits and increment respective index content by 1;
	for(int i = 0; i< n; i++)
	{
		++count[(a[i]/pos) % 10]; // find the positional digit at content of a[i]
	}
	
	//Find Prefix sum
	for (int i = 1; i < 10; i++)
	{
		count[i] = count[i] + count[i-1];
	}

	//Map the item in 'a' starting from right to maintain stability. 
	for(int i = n-1; i >=0; i--)
	{
		b[--count[(a[i]/pos)%10]] = a[i];
	}

	for(int i = 0; i< n; i++)
	{
		a[i] = b[i];
	}
	free(b);
	return retval;
}

uint16_t get_max_item_count(uint16_t*a, uint16_t n)
{
	uint16_t max_item = a[0];
	if (NULL != a)
	{
        for (int i = 1; i < n; i++)
            if (a[i] > max_item)
		    {
                max_item = a[i];
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
	char val = ' ';
	uint16_t key = 0;
	Huffman_Hash_Table* hash_table = NULL;
	Huffman_Pqueue_node* PQueue = NULL;
	create_huffman_hashtable(HashCapacity, &hash_table);

	for(int64_t i = 0; i< bufinlen;i++)
	{
		val = bufin[i];
	    key = hash_function(&val);
	    huff_node_insert(hash_table, &key, &val);
	}
	print_huffman_table(hash_table);
	huffman_hash_table_sort(hash_table, &PQueue);
	free_huffman_hash_table(hash_table);
	return 0;
}


/**
 * TODO Complete this function
 **/
int huffman_decode(const unsigned char *bufin,
						  unsigned int bufinlen,
						  unsigned char **pbufout,
						  unsigned int *pbufoutlen)
{
	return 0;
}
